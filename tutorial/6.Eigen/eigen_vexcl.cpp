#include <vector>
#include <iostream>
#include "util.h"

#include <amgcl/backend/vexcl.hpp>
#include <amgcl/backend/vexcl_static_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/block_matrix.hpp>

#include <amgcl/io/mm.hpp>
#include <amgcl/profiler.hpp>

int main(int argc, char *argv[])
{
    // The command line should contain the matrix file name:
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> <rhs.txt> <solution.txt>" << std::endl;
        return 1;
    }

#pragma region VexCL setup
    // Create VexCL context. Set the environment variable OCL_DEVICE to
    // control which GPU to use in case multiple are available,
    // and use single device:
    vex::Context ctx(vex::Filter::Env && vex::Filter::Count(1));
    std::cout << ctx << std::endl;

    // Enable support for block-valued matrices in the VexCL kernels:
    vex::scoped_program_header h1(ctx, amgcl::backend::vexcl_static_matrix_declaration<double, 3>());
    vex::scoped_program_header h2(ctx, amgcl::backend::vexcl_static_matrix_declaration<float, 3>());
#pragma endregion

    // The profiler:
    amgcl::profiler<> prof("Eigen (VexCL)");

#pragma region Matrix setup
    // Read the system matrix:
    ptrdiff_t rows, cols;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val;

    prof.tic("read");
    std::tie(rows, cols) = amgcl::io::mm_reader(argv[1])(ptr, col, val);
    std::cout << "Matrix " << argv[1] << ": " << rows << "x" << cols << std::endl;
    prof.toc("read");

    // We use the tuple of CRS arrays to represent the system matrix.
    // Note that std::tie creates a tuple of references, so no data is actually
    // copied here:
    auto A = std::tie(rows, ptr, col, val);
#pragma endregion

    // The RHS is loaded from file:
    std::vector<double> f = load_rhs<double>(argv[2], rows);

    /*// Scale the matrix so that it has the unit diagonal.
    // First, find the diagonal values:
    std::vector<double> D(rows, 1.0);
    for(ptrdiff_t i = 0; i < rows; ++i) {
        for(ptrdiff_t j = ptr[i], e = ptr[i+1]; j < e; ++j) {
            if (col[j] == i) {
                D[i] = 1 / sqrt(val[j]);
                break;
            }
        }
    }

    // Then, apply the scaling in-place:
    for(ptrdiff_t i = 0; i < rows; ++i) {
        for(ptrdiff_t j = ptr[i], e = ptr[i+1]; j < e; ++j) {
            val[j] *= D[i] * D[col[j]];
        }
        f[i] *= D[i];
    }*/

#pragma region Solver setup
    // Compose the solver type
    typedef amgcl::static_matrix<double, 3, 3> dmat_type; // matrix value type in double precision
    typedef amgcl::static_matrix<double, 3, 1> dvec_type; // the corresponding vector value type
    typedef amgcl::static_matrix<float, 3, 3> smat_type;  // matrix value type in single precision

    typedef amgcl::backend::vexcl<dmat_type> SBackend; // the solver backend
    typedef amgcl::backend::vexcl<smat_type> PBackend; // the preconditioner backend

    typedef amgcl::make_solver<
        amgcl::amg<
            PBackend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0>,
        amgcl::solver::bicgstab<SBackend>>
        Solver;

    // Solver parameters
    Solver::params params;
    params.solver.maxiter = 500;

    // Set the VexCL context in the backend parameters
    SBackend::params be_params;
    be_params.q = ctx;

    // Initialize the solver with the system matrix.
    // We use the block_matrix adapter to convert the matrix into the block
    // format on the fly:
    prof.tic("setup");
    auto Ab = amgcl::adapter::block_matrix<dmat_type>(A);
    Solver solve(Ab, params, be_params);
    prof.toc("setup");

    // Show the mini-report on the constructed solver:
    std::cout << solve << std::endl;
#pragma endregion

    // Solve the system with the zero initial approximation:
    int iters;
    double error;
    std::vector<double> x(rows, 0.0);

    // Since we are using mixed precision, we have to transfer the system matrix to the GPU:
    prof.tic("GPU matrix");
    auto A_gpu = SBackend::copy_matrix(
        std::make_shared<amgcl::backend::crs<dmat_type>>(Ab), be_params);
    prof.toc("GPU matrix");

    // We reinterpret both the RHS and the solution vectors as block-valued,
    // and copy them to the VexCL vectors:
    auto f_ptr = reinterpret_cast<dvec_type *>(f.data());
    auto x_ptr = reinterpret_cast<dvec_type *>(x.data());
    vex::vector<dvec_type> F(ctx, rows / 3, f_ptr);
    vex::vector<dvec_type> X(ctx, rows / 3, x_ptr);

    prof.tic("solve");
    std::tie(iters, error) = solve(*A_gpu, F, X);
    prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    std::cout << "Iters: " << iters << std::endl
              << "Error: " << error << std::endl
              << prof << std::endl;

    std::cout << "Saving solution..." << std::endl;

    std::vector<dvec_type> solution(X.size());
    X.read_data(0, X.size() * sizeof(dvec_type), solution.data(), true);
    save_solution<double>(argv[3], solution, rows);
}
