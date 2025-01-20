FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS build
RUN apt update
RUN apt install -y cmake g++ \
libboost-test-dev libboost-program-options-dev libboost-serialization-dev libopenmpi-dev

COPY . /amgcl

RUN mkdir /amgcl/build
WORKDIR /amgcl/build

RUN cmake .. -DAMGCL_BUILD_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_TARGET_ARCH=Volta
RUN cmake --build . --config Release -j 8

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime
COPY --from=build /amgcl/build /amgcl
RUN apt update
RUN apt install -y libgomp1 openmpi-bin openmpi-common
