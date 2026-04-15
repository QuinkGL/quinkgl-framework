FROM python:3.10-slim

WORKDIR /app

# Sistem bağımlılıklarını kur
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsodium-dev \
    && rm -rf /var/lib/apt/lists/*

# Python bağımlılıklarını kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY protos ./protos
COPY src ./src
COPY scripts ./scripts

# Protobuf dosyalarını derle
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

# Set PYTHONPATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

CMD ["python3", "scripts/run_gossip_node.py", "--help"]
