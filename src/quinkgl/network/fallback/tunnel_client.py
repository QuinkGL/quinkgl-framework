"""
Tunnel Client for NAT Traversal

Connects to tunnel server via reverse tunnel and relays weight updates.
"""

import asyncio
import logging
import time
from typing import Optional, Callable

import grpc
from quinkgl.network.fallback import tunnel_pb2, tunnel_pb2_grpc

logger = logging.getLogger(__name__)

class TunnelClient:
    """Client for reverse tunnel NAT traversal."""
    
    def __init__(self, tunnel_server: str, node_id: str):
        """
        Initialize tunnel client.
        
        Args:
            tunnel_server: "host:port" of tunnel server
            node_id: Unique identifier for this node
        """
        self.tunnel_server = tunnel_server
        self.node_id = node_id
        self.channel = None
        self.stub = None
        self.message_queue = asyncio.Queue()
        self.running = False
        self.on_chat_message: Optional[Callable] = None
        self.on_peer_list: Optional[Callable] = None
        
        # Signaling callbacks
        self.on_sdp_offer: Optional[Callable] = None
        self.on_sdp_answer: Optional[Callable] = None
        self.on_ice_candidate: Optional[Callable] = None
    
    async def connect(self):
        """Connect to tunnel server."""
        logger.info(f"Connecting to tunnel server {self.tunnel_server}...")
        
        self.channel = grpc.aio.insecure_channel(
            self.tunnel_server,
            options=[
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 60000),
                ('grpc.keepalive_timeout_ms', 20000),
            ]
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
            
            async for msg in self.stub.RegisterTunnel(request_generator()):
                await self._handle_tunnel_message(msg)
        
        except Exception as e:
            logger.error(f"Tunnel stream error: {e}")
            self.running = False
    
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
        
        await self.message_queue.put(msg)
    
    async def _handle_tunnel_message(self, msg: tunnel_pb2.TunnelMessage):
        """Handle incoming tunnel message."""
        if msg.type == tunnel_pb2.TEXT_MESSAGE:
            # Deserialize chat message
            chat_msg = tunnel_pb2.ChatMessage()
            chat_msg.ParseFromString(msg.payload)
            
            if self.on_chat_message:
                await self.on_chat_message(chat_msg)
        
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
            
            await self.message_queue.put(response)
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
        
        await self.message_queue.put(msg)
    
    async def send_sdp_offer(self, target_id: str, offer_payload: bytes):
        """Send SDP offer to peer."""
        msg = tunnel_pb2.TunnelMessage(
            node_id=self.node_id,
            target_id=target_id,
            type=tunnel_pb2.SDP_OFFER,
            payload=offer_payload,
            timestamp=int(time.time() * 1000)
        )
        await self.message_queue.put(msg)

    async def send_sdp_answer(self, target_id: str, answer_payload: bytes):
        """Send SDP answer to peer."""
        msg = tunnel_pb2.TunnelMessage(
            node_id=self.node_id,
            target_id=target_id,
            type=tunnel_pb2.SDP_ANSWER,
            payload=answer_payload,
            timestamp=int(time.time() * 1000)
        )
        await self.message_queue.put(msg)

    async def send_ice_candidate(self, target_id: str, candidate_payload: bytes):
        """Send ICE candidate to peer."""
        msg = tunnel_pb2.TunnelMessage(
            node_id=self.node_id,
            target_id=target_id,
            type=tunnel_pb2.ICE_CANDIDATE,
            payload=candidate_payload,
            timestamp=int(time.time() * 1000)
        )
        await self.message_queue.put(msg)
    
    async def close(self):
        """Close tunnel connection."""
        self.running = False
        if self.channel:
            await self.channel.close()
        logger.info("Tunnel client closed")
