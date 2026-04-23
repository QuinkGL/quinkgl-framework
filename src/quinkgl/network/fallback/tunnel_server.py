#!/usr/bin/env python3
"""
Tunnel Server for NAT Traversal

Relays traffic between clients behind NAT using gRPC bidirectional streaming.
"""

import asyncio
import logging
import argparse
import time
from typing import Dict, Optional
from concurrent import futures
from datetime import datetime, timedelta

import grpc
from quinkgl.network.fallback import tunnel_pb2, tunnel_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Server-side limits
MAX_TUNNELS = 500
MAX_SIGNALING_SESSIONS = 1000
SIGNALING_SESSION_TIMEOUT = timedelta(minutes=10)
PER_CLIENT_QUEUE_MAXSIZE = 256


class TunnelServicer(tunnel_pb2_grpc.TunnelServiceServicer):
    """gRPC tunnel service implementation."""
    
    def __init__(self):
        self.tunnels: Dict[str, asyncio.Queue] = {}  # node_id -> message queue
        self.last_seen: Dict[str, datetime] = {}
        self.timeout = timedelta(minutes=5)
        
        # Signaling state (for P2P)
        self.signaling_sessions: Dict[str, Dict] = {}  # session_id -> {peer_a, peer_b, state}
    
    async def RegisterTunnel(self, request_iterator, context):
        """Handle bidirectional tunnel stream."""
        node_id = None
        # B17 §6.4: Bounded per-client queue
        message_queue = asyncio.Queue(maxsize=PER_CLIENT_QUEUE_MAXSIZE)
        
        try:
            # Handle incoming messages
            async def handle_incoming():
                nonlocal node_id
                try:
                    async for msg in request_iterator:
                        try:
                            if msg.type == tunnel_pb2.REGISTER:
                                # B17 §6.1: Reject duplicate registrations
                                if msg.node_id in self.tunnels:
                                    logger.warning(
                                        f"Duplicate REGISTER for '{msg.node_id}' — rejected"
                                    )
                                    continue

                                # B17 §6.2: Enforce tunnel capacity
                                if len(self.tunnels) >= MAX_TUNNELS:
                                    logger.warning(
                                        f"Tunnel capacity reached ({MAX_TUNNELS}) — "
                                        f"rejecting '{msg.node_id}'"
                                    )
                                    continue

                                node_id = msg.node_id
                                self.tunnels[node_id] = message_queue
                                self.last_seen[node_id] = datetime.now()
                                logger.info(f"✓ Registered tunnel for node '{node_id}'")

                                # B17 §6.3: Incremental peer notification
                                # Send full peer list only to the NEW node.
                                # Send a single-element diff (new-peer-joined) to existing nodes.
                                peer_ids = list(self.tunnels.keys())
                                ts_now = int(time.time() * 1000)

                                # New node gets full list
                                others = [p for p in peer_ids if p != node_id]
                                new_peer_list = tunnel_pb2.PeerListPayload(peer_ids=others)
                                await message_queue.put(tunnel_pb2.TunnelMessage(
                                    node_id="server",
                                    target_id=node_id,
                                    type=tunnel_pb2.PEER_LIST,
                                    payload=new_peer_list.SerializeToString(),
                                    timestamp=ts_now,
                                ))

                                # Existing nodes get incremental diff
                                diff_list = tunnel_pb2.PeerListPayload(peer_ids=[node_id])
                                for pid, queue in self.tunnels.items():
                                    if pid == node_id:
                                        continue
                                    try:
                                        queue.put_nowait(tunnel_pb2.TunnelMessage(
                                            node_id="server",
                                            target_id=pid,
                                            type=tunnel_pb2.PEER_LIST,
                                            payload=diff_list.SerializeToString(),
                                            timestamp=ts_now,
                                        ))
                                    except asyncio.QueueFull:
                                        logger.warning(f"Queue full for {pid}, skipping diff")

                            elif node_id is None:
                                logger.warning("Received tunnel message before REGISTER — ignored")
                                continue

                            elif msg.node_id != node_id:
                                logger.warning(
                                    f"Rejected tunnel message with spoofed sender: stream={node_id} payload={msg.node_id}"
                                )
                                continue

                            elif msg.type == tunnel_pb2.HEARTBEAT:
                                if msg.node_id in self.last_seen:
                                    self.last_seen[msg.node_id] = datetime.now()
                            
                            elif msg.type == tunnel_pb2.TEXT_MESSAGE:
                                target_id = msg.target_id
                                if target_id in self.tunnels:
                                    logger.info(f"Relaying message: {msg.node_id} → {target_id}")
                                    await self.tunnels[target_id].put(msg)
                                else:
                                    logger.warning(f"Target '{target_id}' not found")
                                    error = tunnel_pb2.ErrorPayload(
                                        code="TARGET_NOT_FOUND",
                                        message=f"Peer '{target_id}' is not connected"
                                    )
                                    error_msg = tunnel_pb2.TunnelMessage(
                                        node_id="server",
                                        target_id=msg.node_id,
                                        type=tunnel_pb2.ERROR,
                                        payload=error.SerializeToString(),
                                        timestamp=int(time.time() * 1000)
                                    )
                                    await message_queue.put(error_msg)
                            
                            # NEW: Signaling messages for P2P
                            elif msg.type == tunnel_pb2.SDP_OFFER:
                                # Relay SDP offer to target peer
                                target_id = msg.target_id
                                if target_id in self.tunnels:
                                    logger.info(f"Relaying SDP offer: {msg.node_id} → {target_id}")
                                    await self.tunnels[target_id].put(msg)
                                else:
                                    logger.warning(f"SDP offer target '{target_id}' not found")
                            
                            elif msg.type == tunnel_pb2.SDP_ANSWER:
                                # Relay SDP answer to target peer
                                target_id = msg.target_id
                                if target_id in self.tunnels:
                                    logger.info(f"Relaying SDP answer: {msg.node_id} → {target_id}")
                                    await self.tunnels[target_id].put(msg)
                                else:
                                    logger.warning(f"SDP answer target '{target_id}' not found")
                            
                            elif msg.type == tunnel_pb2.ICE_CANDIDATE:
                                # Relay ICE candidate to target peer
                                target_id = msg.target_id
                                if target_id in self.tunnels:
                                    logger.debug(f"Relaying ICE candidate: {msg.node_id} → {target_id}")
                                    await self.tunnels[target_id].put(msg)
                                else:
                                    logger.warning(f"ICE candidate target '{target_id}' not found")
                        except Exception as e:
                            logger.error(f"Error handling message from {node_id}: {e}")
                            # Continue loop
                            
                except Exception as e:
                    logger.error(f"Fatal error in incoming handler for {node_id}: {e}")
                    raise

            # Start incoming handler
            incoming_task = asyncio.create_task(handle_incoming())
            
            # Send outgoing messages
            while not context.cancelled():
                if incoming_task.done():
                    break
                
                try:
                    # Wait for message or incoming task completion
                    get_msg_task = asyncio.create_task(message_queue.get())
                    done, pending = await asyncio.wait(
                        [get_msg_task, incoming_task],
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=30.0
                    )
                    
                    if incoming_task in done:
                        get_msg_task.cancel()
                        break
                    
                    if get_msg_task in done:
                        try:
                            msg = get_msg_task.result()
                            yield msg
                        except Exception:
                            # Task failed, continue to next iteration
                            continue
                    else:
                        # Timeout - send heartbeat
                        if node_id:
                            heartbeat = tunnel_pb2.TunnelMessage(
                                node_id="server",
                                target_id=node_id,
                                type=tunnel_pb2.HEARTBEAT,
                                timestamp=int(time.time() * 1000)
                            )
                            yield heartbeat
                            
                except Exception as e:
                    logger.error(f"Error in outgoing loop for {node_id}: {e}")
                    break
        
        except Exception as e:
            logger.error(f"Tunnel error for '{node_id}': {e}")
        
        finally:
            # Cleanup
            if node_id and node_id in self.tunnels:
                del self.tunnels[node_id]
                del self.last_seen[node_id]
                logger.info(f"✗ Tunnel closed for node '{node_id}'")
                
                # Notify other peers
                peer_ids = list(self.tunnels.keys())
                for pid, queue in self.tunnels.items():
                    others = [p for p in peer_ids if p != pid]
                    
                    peer_list = tunnel_pb2.PeerListPayload(peer_ids=others)
                    update = tunnel_pb2.TunnelMessage(
                        node_id="server",
                        target_id=pid,
                        type=tunnel_pb2.PEER_LIST,
                        payload=peer_list.SerializeToString(),
                        timestamp=int(time.time() * 1000)
                    )
                    await queue.put(update)
            
            incoming_task.cancel()
    
    async def cleanup_stale_tunnels(self):
        """Remove tunnels that haven't sent heartbeat."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            now = datetime.now()
            stale = [
                node_id for node_id, last_seen in self.last_seen.items()
                if now - last_seen > self.timeout
            ]
            
            for node_id in stale:
                logger.info(f"✗ Removing stale tunnel: {node_id}")
                if node_id in self.tunnels:
                    del self.tunnels[node_id]
                if node_id in self.last_seen:
                    del self.last_seen[node_id]

            # B17 §6.2: Prune stale signaling sessions
            stale_sessions = [
                sid for sid, sess in self.signaling_sessions.items()
                if now - sess.get("created_at", now) > SIGNALING_SESSION_TIMEOUT
            ]
            for sid in stale_sessions:
                del self.signaling_sessions[sid]
            if stale_sessions:
                logger.info(f"Pruned {len(stale_sessions)} stale signaling sessions")

async def serve(
    host: str,
    port: int,
    # NET-016/017: TLS/mTLS server configuration
    root_certificates_path: Optional[str] = None,
    private_key_path: Optional[str] = None,
    certificate_chain_path: Optional[str] = None,
    require_client_cert: bool = False,
):
    """Start the tunnel server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        root_certificates_path: Path to CA cert PEM (for TLS/mTLS).
        private_key_path: Path to server key PEM (for TLS/mTLS).
        certificate_chain_path: Path to server cert PEM (for TLS/mTLS).
        require_client_cert: If True, require client certificates (mTLS).
    """
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 60000),
            ('grpc.keepalive_timeout_ms', 20000),
        ]
    )
    
    servicer = TunnelServicer()
    tunnel_pb2_grpc.add_TunnelServiceServicer_to_server(servicer, server)
    
    listen_addr = f'{host}:{port}'

    # NET-016/017: Use secure port when TLS credentials are configured
    if private_key_path and certificate_chain_path:
        with open(private_key_path, 'rb') as f:
            server_key = f.read()
        with open(certificate_chain_path, 'rb') as f:
            server_cert = f.read()

        root_certificates = None
        if root_certificates_path:
            with open(root_certificates_path, 'rb') as f:
                root_certificates = f.read()

        server_credentials = grpc.ssl_server_credentials(
            private_key_certificate_chain_pairs=[(server_key, server_cert)],
            root_certificates=root_certificates,
            require_client_auth=require_client_cert,
        )
        server.add_secure_port(listen_addr, server_credentials)
        logger.info(f"TLS/mTLS enabled (require_client_cert={require_client_cert})")
    else:
        server.add_insecure_port(listen_addr)
        logger.info("Running in insecure mode (no TLS)")
    
    await server.start()
    logger.info(f"✓ Tunnel server started on {listen_addr}")
    logger.info("Waiting for client tunnels...")
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(servicer.cleanup_stale_tunnels())
    
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        cleanup_task.cancel()
        await server.stop(grace=5)

def main():
    parser = argparse.ArgumentParser(description="QuinkGL Tunnel Server for NAT Traversal")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=50051, help="Port to bind to")
    # NET-016/017: TLS/mTLS CLI arguments
    parser.add_argument("--root-cert", type=str, default=None, help="Path to CA cert PEM")
    parser.add_argument("--server-key", type=str, default=None, help="Path to server key PEM")
    parser.add_argument("--server-cert", type=str, default=None, help="Path to server cert PEM")
    parser.add_argument("--require-client-cert", action="store_true", help="Require client certs (mTLS)")
    
    args = parser.parse_args()
    
    asyncio.run(serve(
        args.host,
        args.port,
        root_certificates_path=args.root_cert,
        private_key_path=args.server_key,
        certificate_chain_path=args.server_cert,
        require_client_cert=args.require_client_cert,
    ))

if __name__ == "__main__":
    main()
