# squid5090 server image. Build from the repo root: wslc build -f deploy/Dockerfile -t squid5090 .
FROM ubuntu:24.04 AS cuda-apt
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg xz-utils \
 && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-archive-keyring.gpg -o /usr/share/keyrings/cuda.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /" > /etc/apt/sources.list.d/cuda.list \
 && apt-get update

FROM cuda-apt AS build
RUN apt-get install -y --no-install-recommends build-essential cuda-nvcc-13-1=13.1.115-1 libcublas-dev-13-1
RUN curl -fsSL https://ziglang.org/download/0.17.1/zig-x86_64-linux-0.17.1.tar.xz -o /tmp/zig.tar.xz \
 && echo "3581863e3cbcfd0fce0efc682ce7e9e93da96668a977791d356a26d2a7e9b17a  /tmp/zig.tar.xz" | sha256sum -c \
 && mkdir /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz
WORKDIR /src
COPY build.zig ./
COPY cuda cuda
COPY src src
COPY third_party third_party
# ponytail: prefill batch is a source constant upstream; sed it here, make it a -D option if it needs more values
ARG PREFILL_BATCH=2048
RUN sed -i "s/^const prefill_batch_max: u32 = 2048;/const prefill_batch_max: u32 = ${PREFILL_BATCH};/" src/engine.zig && grep -q "^const prefill_batch_max: u32 = ${PREFILL_BATCH};" src/engine.zig
RUN ls -d /usr/local/cuda-13.1/lib64 && /opt/zig/zig build -Dcuda-path=/usr/local/cuda-13.1 --summary all
RUN mkdir /out && cp zig-out/bin/squid5090 /out/ && cp -P /usr/local/cuda-13.1/lib64/libcudart.so* /usr/local/cuda-13.1/lib64/libcublas.so* /usr/local/cuda-13.1/lib64/libcublasLt.so* /out/

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /opt/squid/
ENV LD_LIBRARY_PATH=/opt/squid:/usr/lib/wsl/lib
EXPOSE 8000
ENTRYPOINT ["/opt/squid/squid5090"]
