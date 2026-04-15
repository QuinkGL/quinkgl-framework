# QuinkGL Tunnel Server

NAT traversal relay sunucusu - P2P bağlantı kuramayan node'lar için fallback mekanizması.

## Ne İşe Yarar?

```
┌─────────────────────────────────────────────────────────────────┐
│                         INTERNET                                │
│                                                                 │
│   ┌─────────┐         ┌─────────────────┐         ┌─────────┐  │
│   │  Node A │◄───────►│  Tunnel Server  │◄───────►│  Node B │  │
│   │ (NAT)   │         │ (Public IP)     │         │ (NAT)   │  │
│   └─────────┘         └─────────────────┘         └─────────┘  │
│                                                                 │
│   NAT arkasındaki node'lar birbirine doğrudan bağlanamaz.       │
│   Tunnel server, mesajları relay ederek bağlantı sağlar.        │
└─────────────────────────────────────────────────────────────────┘
```

## Kullanım

### Lokal Test

```bash
python tunnel_server.py --host 0.0.0.0 --port 50051
```

### Production (Cloud)

```bash
# Docker ile
docker build -t quinkgl-tunnel .
docker run -p 50051:50051 quinkgl-tunnel

# Veya doğrudan
python tunnel_server.py --host 0.0.0.0 --port 50051
```

### Client Tarafı

Node'lar tunnel server'a bağlanmak için:

```python
from quinkgl.network.tunnel_client import TunnelClient

client = TunnelClient(
    tunnel_host="your-server.com",
    tunnel_port=50051,
    node_id="alice"
)
await client.connect()
```

## Deployment

### AWS/GCP/Azure

1. Bir VM oluştur (t2.micro yeterli)
2. Port 50051'i aç (Security Group/Firewall)
3. `tunnel_server.py`'ı çalıştır

### Docker Compose

```yaml
version: '3'
services:
  tunnel:
    build: .
    ports:
      - "50051:50051"
    restart: always
```

## Protokol

gRPC bidirectional streaming kullanır:
- `RegisterTunnel`: Node'ları kaydeder
- Heartbeat: 30 saniyede bir
- Stale cleanup: 60 saniye timeout

## Dosyalar

- `tunnel_server.py` - Ana sunucu kodu
- `tunnel-deploy.tar.gz` - Deployment paketi
