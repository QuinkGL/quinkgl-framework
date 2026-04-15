"""
Fallback Network Module

Provides tunnel-based relay as fallback when IPv8 P2P fails.
Used for NAT traversal in restrictive network environments.
"""

from quinkgl.network.fallback.tunnel_client import TunnelClient
from quinkgl.network.fallback.tunnel_pb2 import *
from quinkgl.network.fallback.tunnel_pb2_grpc import *
from quinkgl.network.fallback.tunnel_server import TunnelServicer, serve

__all__ = [
    "TunnelClient",
    "TunnelServicer",
    "serve",
]
