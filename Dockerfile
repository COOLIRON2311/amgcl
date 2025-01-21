FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS build
RUN apt update
RUN apt install -y --no-install-recommends cmake g++ \
libboost-test-dev libboost-program-options-dev libboost-serialization-dev libopenmpi-dev \
libboost-chrono-dev libboost-date-time-dev libboost-filesystem-dev libboost-thread-dev opencl-clhpp-headers

WORKDIR /amgcl
COPY amgcl amgcl
COPY cmake cmake
COPY docs docs
COPY examples examples
COPY lib lib
COPY tests tests
COPY tutorial tutorial
COPY vexcl vexcl
COPY CMakeLists.txt .

WORKDIR /amgcl/vexcl/build
RUN cmake .. -DCUDA_CUDA_LIBRARY=/usr/local/cuda/compat/libcuda.so -DCMAKE_BUILD_TYPE=Release
RUN cmake --build . --config Release -j 4
RUN cmake --build . --target install --config Release

WORKDIR /amgcl/build
RUN cmake .. -DAMGCL_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_TARGET_ARCH=Volta
RUN cmake --build . --config Release -j 4

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

RUN apt update
RUN apt install -y --no-install-recommends libgomp1 openmpi-bin openmpi-common \
libboost-test-dev libboost-program-options-dev libboost-serialization-dev libopenmpi-dev \
libboost-chrono-dev libboost-date-time-dev libboost-filesystem-dev libboost-thread-dev

COPY --from=build /amgcl/build/examples /examples
COPY --from=build /amgcl/build/tutorial/2.Serena/ /serena
