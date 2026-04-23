"""
Tunnel Client for NAT Traversal

Connects to tunnel server via reverse tunnel and relays weight updates.
"""

import asyncio
import logging
import time
from types import SimpleNamespace
from typing import Optional, Callable

import grpc
from quinkgl.network.fallback import tunnel_pb2, tunnel_pb2_grpc

logger = logging.getLogger(__name__)

# B10: Reconnect parameters
RECONNECT_INITIAL_DELAY = 1.0   # seconds
RECONNECT_MAX_DELAY = 60.0      # seconds
RECONNECT_BACKOFF_FACTOR = 2.0
RECONNECT_MAX_ATTEMPTS = 10


class TunnelClient:
    """Client for reverse tunnel NAT traversal."""
    
    def __init__(
        self,
        tunnel_server: str,
        node_id: str,
        # NET-016/017: TLS/mTLS configuration
        root_certificates_path: Optional[str] = None,
        private_key_path: Optional[str] = None,
        certificate_chain_path: Optional[str] = None,
        register_deadline_seconds: float = 30.0,
    ):
        """
        Initialize tunnel client.
        
        Args:
            tunnel_server: "host:port" of tunnel server
            node_id: Unique identifier for this node
            root_certificates_path: Path to CA cert PEM (for TLS/mTLS)
            private_key_path: Path to client key PEM (for mTLS)
            certificate_chain_path: Path to client cert PEM (for mTLS)
            register_deadline_seconds: NET-017 per-call deadline for RegisterTunnel
        """
        self.tunnel_server = tunnel_server
        self.node_id = node_id
        self.channel = None
        self.stub = None
        # B9: Bounded queue — prevents memory leak if stream dies
        self.message_queue = asyncio.Queue(maxsize=1024)
        self.running = False
        self.on_chat_message: Optional[Callable] = None
        self.on_peer_list: Optional[Callable] = None
        # B9: Notified when the bidirectional stream dies
        self.on_disconnected: Optional[Callable] = None
        # B10: Reconnect state
        self._reconnect_enabled = True
        self._reconnect_task: Optional[asyncio.Task] = None
        # NET-016/017: TLS config
        self._root_certificates_path = root_certificates_path
        self._private_key_path = private_key_path
        self._certificate_chain_path = certificate_chain_path
        self._register_deadline_seconds = register_deadline_seconds
        
        # Signaling callbacks
        self.on_sdp_offer: Optional[Callable] = None
        self.on_sdp_answer: Optional[Callable] = None
        self.on_ice_candidate: Optional[Callable] = None
    
    def _build_channel_credentials(self) -> Optional[grpc.ChannelCredentials]:
        """NET-016/017: Build gRPC channel credentials for TLS/mTLS."""
        if self._root_certificates_path is None:
            return None  # insecure channel

        with open(self._root_certificates_path, 'rb') as f:
            root_certificates = f.read()

        if self._private_key_path and self._certificate_chain_path:
            # mTLS: client presents certificate
            with open(self._private_key_path, 'rb') as f:
                private_key = f.read()
            with open(self._certificate_chain_path, 'rb') as f:
                certificate_chain = f.read()
            return grpc.ssl_channel_credentials(
                root_certificates=root_certificates,
                private_key=private_key,
                certificate_chain=certificate_chain,
            )
        else:
            # TLS only (server-authenticated)
            return grpc.ssl_channel_credentials(
                root_certificates=root_certificates,
            )

    async def connect(self):
        """Connect to tunnel server."""
        logger.info(f"Connecting to tunnel server {self.tunnel_server}...")

        channel_options = [
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 60000),
            ('grpc.keepalive_timeout_ms', 20000),
        ]

        # NET-016/017: Use secure channel when TLS credentials are configured
        credentials = self._build_channel_credentials()
        if credentials is not None:
            self.channel = grpc.aio.secure_channel(
                self.tunnel_server,
                credentials,
                options=channel_options,
            )
            logger.info("Using TLS/mTLS secure channel")
        else:
            self.channel = grpc.aio.insecure_channel(
                self.tunnel_server,
                options=channel_options,
            )
        
        self.stub = tunnel_pb2_grpc.TunnelServiceStub(self.channel)
        self.running = True
        
        # Start tunnel stream
        asyncio.create_task(self._tunnel_stream())
        
        # Send registration
        await self._send_register()
        
        logger.info(f"✓ Connected to tunnel server")
    
    async def _tunnel_stream(self):
        """Maintain bidirectional stream with tunnel server."""
        try:
            async def request_generator():
                while self.running:
                    msg = await self.message_queue.get()
                    yield msg
            
            # NET-017: Per-call deadline on RegisterTunnel
            async for msg in self.stub.RegisterTunnel(
                request_generator(),
                timeout=self._register_deadline_seconds,
            ):
                await self._handle_tunnel_message(msg)
        
        except Exception as e:
            logger.error(f"Tunnel stream error: {e}")
            self.running = False
            # B9: Surface stream death to upper layers
            if self.on_disconnected:
                try:
                    await self.on_disconnected()
                except Exception as cb_err:
                    logger.debug(f"on_disconnected callback error: {cb_err}")
            # B10: Attempt reconnection
            if self._reconnect_enabled:
                self._reconnect_task = asyncio.ensure_future(self._reconnect_loop())
    
    async def _send_register(self):
        """Send registration message."""
        register_payload = tunnel_pb2.RegisterPayload(
            node_id=self.node_id,
            version="1.0"
        )
        
        msg = tunnel_pb2.TunnelMessage(
            node_id=self.node_id,
            type=tunnel_pb2.REGISTER,
            payload=register_payload.SerializeToString(),
            timestamp=int(time.time() * 1000)
        )

        # NET-032: Add deadline to prevent indefinite blocking
        try:
            await asyncio.wait_for(self.message_queue.put(msg), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Tunnel client message queue put timed out (REGISTER)")
            raise
    
    async def _handle_tunnel_message(self, msg: tunnel_pb2.TunnelMessage):
        """Handle incoming tunnel message."""
        if msg.type == tunnel_pb2.TEXT_MESSAGE:
            # Deserialize chat message
            chat_msg = tunnel_pb2.ChatMessage()
            chat_msg.ParseFromString(msg.payload)
            
            if self.on_chat_message:
                await self.on_chat_message(SimpleNamespace(
                    sender_id=chat_msg.sender_id,
                    text=chat_msg.text,
                    timestamp=chat_msg.timestamp,
                    _tunnel_sender_id=msg.node_id,
                ))
        
        elif msg.type == tunnel_pb2.PEER_LIST:
            # Parse peer list
            peer_list = tunnel_pb2.PeerListPayload()
            peer_list.ParseFromString(msg.payload)
            
            logger.info(f"Available peers: {list(peer_list.peer_ids)}")
            
            if self.on_peer_list:
                await self.on_peer_list(list(peer_list.peer_ids))
        
        elif msg.type == tunnel_pb2.HEARTBEAT:
            # Respond to heartbeat
            heartbeat = tunnel_pb2.HeartbeatPayload(
                node_id=self.node_id,
                timestamp=int(time.time() * 1000)
            )

            response = tunnel_pb2.TunnelMessage(
                node_id=self.node_id,
                type=tunnel_pb2.HEARTBEAT,
                payload=heartbeat.SerializeToString(),
                timestamp=int(time.time() * 1000)
            )

            # NET-032: Add deadline to prevent indefinite blocking
            try:
                await asyncio.wait_for(self.message_queue.put(response), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Tunnel client message queue put timed out (HEARTBEAT)")
                raise
        elif msg.type == tunnel_pb2.ERROR:
            try:
                error = tunnel_pb2.ErrorPayload()
                error.ParseFromString(msg.payload)
                logger.error(f"Tunnel error: {error.code} - {error.message}")
            except Exception as e:
                logger.error(f"Failed to parse error payload: {e}")
            
        # Signaling messages
        elif msg.type == tunnel_pb2.SDP_OFFER:
            logger.info(f"Received SDP_OFFER from {msg.node_id}, payload size: {len(msg.payload)}")
            if self.on_sdp_offer:
                try:
                    offer = tunnel_pb2.SDPOfferPayload()
                    offer.ParseFromString(msg.payload)
                    await self.on_sdp_offer(offer)
                except Exception as e:
                    logger.error(f"Error handling SDP_OFFER: {e}")
            else:
                logger.warning("No handler for SDP_OFFER")
                
        elif msg.type == tunnel_pb2.SDP_ANSWER:
            logger.info(f"Received SDP_ANSWER from {msg.node_id}")
            if self.on_sdp_answer:
                try:
                    answer = tunnel_pb2.SDPAnswerPayload()
                    answer.ParseFromString(msg.payload)
                    await self.on_sdp_answer(answer)
                except Exception as e:
                    logger.error(f"Error handling SDP_ANSWER: {e}")
                
        elif msg.type == tunnel_pb2.ICE_CANDIDATE:
            if self.on_ice_candidate:
                try:
                    candidate = tunnel_pb2.ICECandidatePayload()
                    candidate.ParseFromString(msg.payload)
                    await self.on_ice_candidate(msg.node_id, candidate)
                except Exception as e:
                    logger.error(f"Error handling ICE_CANDIDATE: {e}")
    
    async def send_chat_message(self, target_id: str, text: str):
        """Send chat message to peer via tunnel."""
        chat_msg = tunnel_pb2.ChatMessage(
            sender_id=self.node_id,
            text=text,
            timestamp=int(time.time() * 1000)
        )
        
        msg = tunnel_pb2.TunnelMessage(
            node_id=self.node_id,
            target_id=target_id,
            type=tunnel_pb2.TEXT_MESSAGE,
            payload=chat_msg.SerializeToString(),
            timestamp=int(time.time() * 1000)
        )

        # NET-032: Add deadline to prevent indefinite blocking
        try:
            await asyncio.wait_for(self.message_queue.put(msg), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Tunnel client message queue put timed out (TEXT_MESSAGE)")
            raise
    
    async def send_sdp_offer(self, target_id: str, offer_payload: bytes):
        """Send SDP offer to peer."""
        msg = tunnel_pb2.TunnelMessage(
            node_id=self.node_id,
            target_id=target_id,
            type=tunnel_pb2.SDP_OFFER,
            payload=offer_payload,
            timestamp=int(time.time() * 1000)
        )
        # NET-032: Add deadline to prevent indefinite blocking
        try:
            await asyncio.wait_for(self.message_queue.put(msg), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Tunnel client message queue put timed out (SDP_OFFER)")
            raise

    async def send_sdp_answer(self, target_id: str, answer_payload: bytes):
        """Send SDP answer to peer."""
        msg = tunnel_pb2.TunnelMessage(
            node_id=self.node_id,
            target_id=target_id,
            type=tunnel_pb2.SDP_ANSWER,
            payload=answer_payload,
            timestamp=int(time.time() * 1000)
        )
        # NET-032: Add deadline to prevent indefinite blocking
        try:
            await asyncio.wait_for(self.message_queue.put(msg), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Tunnel client message queue put timed out (SDP_ANSWER)")
            raise

    async def send_ice_candidate(self, target_id: str, candidate_payload: bytes):
        """Send ICE candidate to peer."""
        msg = tunnel_pb2.TunnelMessage(
            node_id=self.node_id,
            target_id=target_id,
            type=tunnel_pb2.ICE_CANDIDATE,
            payload=candidate_payload,
            timestamp=int(time.time() * 1000)
        )
        # NET-032: Add deadline to prevent indefinite blocking
        try:
            await asyncio.wait_for(self.message_queue.put(msg), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Tunnel client message queue put timed out (ICE_CANDIDATE)")
            raise
    
    async def _reconnect_loop(self):
        """B10: Reconnect with exponential backoff after stream failure."""
        delay = RECONNECT_INITIAL_DELAY
        for attempt in range(1, RECONNECT_MAX_ATTEMPTS + 1):
            if not self._reconnect_enabled:
                return
            logger.info(
                f"Tunnel reconnect attempt {attempt}/{RECONNECT_MAX_ATTEMPTS} "
                f"in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)
            try:
                # Reset queue for fresh connection
                self.message_queue = asyncio.Queue(maxsize=1024)
                await self.connect()
                logger.info("Tunnel reconnected successfully")
                return
            except Exception as e:
                logger.warning(f"Reconnect attempt {attempt} failed: {e}")
            delay = min(delay * RECONNECT_BACKOFF_FACTOR, RECONNECT_MAX_DELAY)
        logger.error(
            f"Tunnel reconnect failed after {RECONNECT_MAX_ATTEMPTS} attempts — giving up"
        )

    async def close(self):
        """Close tunnel connection."""
        self.running = False
        self._reconnect_enabled = False
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
        if self.channel:
            await self.channel.close()
        logger.info("Tunnel client closed")
