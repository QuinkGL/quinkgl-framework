# QuinkGL Configuration
import os
import logging

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass 

def _validate_host_port(host_port: str) -> tuple[str, str]:
    """
    Validate and split a host:port string.

    Args:
        host_port: String in format "host:port"

    Returns:
        Tuple of (host, port) as strings

    Raises:
        ValueError: If format is invalid or port out of range
    """
    if not host_port:
        raise ValueError("Host:port string cannot be empty")

    if host_port.startswith('['):
        if ']:' not in host_port:
            raise ValueError(f"Invalid IPv6 format, got: {host_port}")
        host_part, port = host_port.rsplit(':', 1)
        host = host_part.lstrip('[').rstrip(']')
    else:
        parts = host_port.rsplit(':', 1)
        if len(parts) != 2:
            raise ValueError(f"Must be in 'host:port' format, got: {host_port}")
        host, port = parts

    if not host:
        raise ValueError("Host cannot be empty")

    if not port.isdigit():
        raise ValueError(f"Port must be numeric, got: {port}")

    port_num = int(port)
    if not (1 <= port_num <= 65535):
        raise ValueError(f"Port must be 1-65535, got: {port_num}")

    return host, port

# Public Fallback Server
PRODUCTION_TUNNEL_SERVER = "quinkql-tunnel.duckdns.org:50051"

def _get_tunnel_server() -> str:
    """Get TUNNEL_SERVER from environment with fallback to production server."""
    if server := os.getenv("TUNNEL_SERVER"):
        return server
    return PRODUCTION_TUNNEL_SERVER


# Tunnel Configuration (Fallback for IPv8 P2P)
# IPv8 handles NAT traversal directly, tunnel is used only as fallback
# TUNNEL_SERVER can be set via environment variable or .env file
# Defaults to localhost:50051 for local development
_TUNNEL_SERVER = _get_tunnel_server()

_TUNNEL_HOST, _TUNNEL_PORT = _validate_host_port(_TUNNEL_SERVER)
logger.info(f"Tunnel server configured: {_TUNNEL_HOST}:{_TUNNEL_PORT}")

TUNNEL_CONFIG = {
    "server": _TUNNEL_SERVER,
    "host": _TUNNEL_HOST,
    "port": int(_TUNNEL_PORT),
    "timeout": 30,
    "heartbeat_interval": 30,
}

# Gossip Configuration
GOSSIP_CONFIG = {
    "publish_interval": 10,  # seconds
    "max_peers": 10,
}
