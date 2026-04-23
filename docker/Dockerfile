# T-DOPS-05: Python 3.12, digest-pinned base, pip install .[all]
FROM python:3.12-slim@sha256:67178d1a226c3c28e8e0f5891b1bc362a6ab38f8f3d5864b7b9ee5e8c45e9d20

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsodium-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for layer caching
COPY pyproject.toml MANIFEST.in ./
COPY protos ./protos
COPY src ./src

# Install the package with all extras (including dev for grpcio-tools at build time)
RUN pip install --no-cache-dir .[all]

# Compile protobuf files (grpcio-tools is a dev extra)
RUN python3 -m grpc_tools.protoc \
    -I./protos \
    --python_out=./src/quinkgl/network \
    --grpc_python_out=./src/quinkgl/network \
    ./protos/gossip.proto && \
    python3 -m grpc_tools.protoc \
    -I./protos \
    --python_out=./src/quinkgl/network \
    --grpc_python_out=./src/quinkgl/network \
    ./protos/tunnel.proto

# Fix imports in generated protobuf files
RUN sed -i 's/^import tunnel_pb2/from quinkgl.network import tunnel_pb2/' src/quinkgl/network/tunnel_pb2_grpc.py && \
    sed -i 's/^import gossip_pb2/from quinkgl.network import gossip_pb2/' src/quinkgl/network/gossip_pb2_grpc.py

# Copy scripts
COPY scripts ./scripts

# Set PYTHONPATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Create non-root user
RUN useradd -m -u 1000 quinkgl && \
    chown -R quinkgl:quinkgl /app
USER quinkgl

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Set entrypoint
ENTRYPOINT ["python3"]

CMD ["scripts/run_gossip_node.py", "--help"]
