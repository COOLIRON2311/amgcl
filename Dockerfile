FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS build
RUN apt update
RUN apt install -y cmake g++ \
libboost-test-dev libboost-program-options-dev libboost-serialization-dev libopenmpi-dev \
libboost-chrono-dev libboost-date-time-dev libboost-filesystem-dev libboost-thread-dev opencl-clhpp-headers

COPY . /amgcl

RUN mkdir /amgcl/vexcl/build
WORKDIR /amgcl/vexcl/build
RUN cmake .. -DCUDA_CUDA_LIBRARY=/usr/local/cuda/compat/libcuda.so -DCMAKE_BUILD_TYPE=Release
RUN cmake --build . --config Release -j 4
RUN cmake --build . --target install --config Release

RUN mkdir /amgcl/build
WORKDIR /amgcl/build

RUN cmake .. -DAMGCL_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_TARGET_ARCH=Volta
RUN cmake --build . --config Release -j 4

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime
RUN apt update
RUN apt install -y libgomp1 openmpi-bin openmpi-common \
libboost-test-dev libboost-program-options-dev libboost-serialization-dev libopenmpi-dev \
libboost-chrono-dev libboost-date-time-dev libboost-filesystem-dev libboost-thread-dev
COPY --from=build /amgcl/build /amgcl
