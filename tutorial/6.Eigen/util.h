#pragma once
#include <fstream>
#include <vector>

template<typename T>
std::vector<T> load_rhs(const char* path, ptrdiff_t rows)
{
    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::invalid_argument("rhs can't be loaded");

    ptrdiff_t n;
    ifs >> n;

    if (n != rows)
        throw std::runtime_error("mismatched number of rows in matrix and rhs vector");

    std::vector<T> rhs(n);
    for (ptrdiff_t i = 0; i < n; i++)
        ifs >> rhs[i];

    return rhs;
}

template<typename T, typename P>
void save_solution(const char* path, const std::vector<P>& sol, ptrdiff_t rows)
{
    std::ofstream ofs(path);
    if (!ofs.is_open())
        throw std::invalid_argument("solution can't be saved");

    const T* buf = reinterpret_cast<const T*>(sol.data());
    ofs << rows << std::endl;
    for (ptrdiff_t i = 0; i < rows; i++)
        ofs << buf[i] << std::endl;
}

template<typename T, typename P>
void save_solution_bin(const char* path, const std::vector<P>& sol, ptrdiff_t rows)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open())
        throw std::invalid_argument("solution can't be saved");

    ofs.write(reinterpret_cast<const char*>(sol.data()), rows * sizeof(T));
}
