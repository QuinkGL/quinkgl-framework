"""
Gossip Protocol

Message types and protocol definitions for P2P gossip learning.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime


class MessageType(IntEnum):
    """Message types in the gossip protocol."""
    # Discovery messages
    DISCOVERY_ANNOUNCE = 0x01
    DISCOVERY_OFFER = 0x02
    DISCOVERY_ACCEPT = 0x03
    DISCOVERY_REJECT = 0x04

    # Model exchange
    MODEL_REQUEST = 0x10
    MODEL_UPDATE = 0x11
    MODEL_PUSH = 0x12
    MODEL_ACK = 0x13

    # Control messages
    HEARTBEAT = 0x20
    PEER_INFO = 0x21
    ERROR = 0x22

    # Consensus / checkpoint
    CHECKPOINT_ANNOUNCE = 0x30

    # Prototype exchange (FedProto)
    PROTOTYPE_EXCHANGE = 0x40


@dataclass
class GossipMessage:
    """
    Base message class for gossip protocol.
    """
    msg_type: MessageType
    sender_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "msg_type": self.msg_type,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GossipMessage":
        """Create message from dictionary."""
        return cls(
            msg_type=MessageType(data["msg_type"]),
            sender_id=data["sender_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            payload=data.get("payload", {})
        )


@dataclass
class DiscoveryAnnounce(GossipMessage):
    """
    Announcement message for peer discovery.

    Peers announce their domain and data schema to find compatible peers.
    Optionally includes a privacy-preserving data fingerprint for
    domain-aware affinity computation.
    """
    domain: str = ""
    data_schema_hash: str = ""
    model_version: str = "1.0.0"
    fingerprint: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.msg_type == 0:
            self.msg_type = MessageType.DISCOVERY_ANNOUNCE

    @classmethod
    def create(
        cls,
        sender_id: str,
        domain: str,
        data_schema_hash: str,
        model_version: str = "1.0.0",
        fingerprint: Optional[Dict[str, Any]] = None
    ) -> "DiscoveryAnnounce":
        return cls(
            msg_type=MessageType.DISCOVERY_ANNOUNCE,
            sender_id=sender_id,
            domain=domain,
            data_schema_hash=data_schema_hash,
            model_version=model_version,
            fingerprint=fingerprint
        )


@dataclass
class ModelUpdateMessage(GossipMessage):
    """
    Message containing model weights for sharing.
    """
    weights: Any = None  # Serialized model weights
    sample_count: int = 0
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    round_number: int = 0

    def __post_init__(self):
        if self.msg_type == 0:
            self.msg_type = MessageType.MODEL_UPDATE

    @classmethod
    def create(
        cls,
        sender_id: str,
        weights: Any,
        sample_count: int,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        round_number: int = 0
    ) -> "ModelUpdateMessage":
        """Create a model update message."""
        return cls(
            msg_type=MessageType.MODEL_UPDATE,
            sender_id=sender_id,
            weights=weights,
            sample_count=sample_count,
            loss=loss,
            accuracy=accuracy,
            round_number=round_number
        )


@dataclass
class HeartbeatMessage(GossipMessage):
    """Heartbeat message to keep connection alive."""
    sequence_number: int = 0

    def __post_init__(self):
        if self.msg_type == 0:
            self.msg_type = MessageType.HEARTBEAT


@dataclass
class CheckpointAnnounceMessage(GossipMessage):
    """
    Checkpoint announcement for consensus detection.

    Peers periodically announce their current training state
    so that convergence consensus can be detected.
    """
    round_number: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    model_version: str = "1.0.0"

    def __post_init__(self):
        if self.msg_type == 0:
            self.msg_type = MessageType.CHECKPOINT_ANNOUNCE

    @classmethod
    def create(
        cls,
        sender_id: str,
        round_number: int,
        loss: float,
        accuracy: float,
        model_version: str = "1.0.0"
    ) -> "CheckpointAnnounceMessage":
        return cls(
            msg_type=MessageType.CHECKPOINT_ANNOUNCE,
            sender_id=sender_id,
            round_number=round_number,
            loss=loss,
            accuracy=accuracy,
            model_version=model_version
        )


@dataclass
class PrototypeMessage(GossipMessage):
    prototypes_json: str = ""

    def __post_init__(self):
        if self.msg_type == 0:
            self.msg_type = MessageType.PROTOTYPE_EXCHANGE

    @classmethod
    def create(cls, sender_id: str, prototypes_json: str) -> "PrototypeMessage":
        return cls(
            msg_type=MessageType.PROTOTYPE_EXCHANGE,
            sender_id=sender_id,
            prototypes_json=prototypes_json,
        )


class GossipProtocol:
    """
    Implements the gossip protocol for message handling.

    Provides methods for creating, validating, and processing
    gossip messages between peers.
    """

    # Maximum message age for validation (1 hour)
    MAX_MESSAGE_AGE_SECONDS = 3600

    def __init__(self, peer_id: str):
        """
        Initialize the gossip protocol.

        Args:
            peer_id: This peer's unique identifier
        """
        self.peer_id = peer_id
        self._sequence_number = 0
        self._lock: Optional[Any] = None  # asyncio.Lock, deferred import

    def _get_lock(self):
        """Lazy import and return asyncio.Lock for thread-safety."""
        if self._lock is None:
            import asyncio
            self._lock = asyncio.Lock()
        return self._lock

    def create_discovery_announce(
        self,
        domain: str,
        data_schema_hash: str,
        model_version: str = "1.0.0"
    ) -> DiscoveryAnnounce:
        """Create a discovery announce message."""
        return DiscoveryAnnounce.create(
            sender_id=self.peer_id,
            domain=domain,
            data_schema_hash=data_schema_hash,
            model_version=model_version
        )

    def create_model_update(
        self,
        weights: Any,
        sample_count: int,
        loss: float = None,
        accuracy: float = None,
        round_number: int = 0
    ) -> ModelUpdateMessage:
        """Create a model update message."""
        return ModelUpdateMessage.create(
            sender_id=self.peer_id,
            weights=weights,
            sample_count=sample_count,
            loss=loss,
            accuracy=accuracy,
            round_number=round_number
        )

    async def create_heartbeat(self) -> HeartbeatMessage:
        """Create a heartbeat message with thread-safe sequence number."""
        lock = self._get_lock()
        async with lock:
            self._sequence_number += 1
            return HeartbeatMessage(
                sender_id=self.peer_id,
                sequence_number=self._sequence_number
            )

    def validate_message(self, message: GossipMessage) -> bool:
        """
        Validate a received message.

        Args:
            message: The message to validate

        Returns:
            True if message is valid
        """
        # Basic validation
        if not message.sender_id:
            return False

        if message.sender_id == self.peer_id:
            # Don't process messages from self
            return False

        # Timestamp validation - reject very old messages
        message_age = datetime.now() - message.timestamp
        if message_age.total_seconds() > self.MAX_MESSAGE_AGE_SECONDS:
            return False

        # Reject messages from the future (clock skew protection)
        if message_age.total_seconds() < -60:  # 1 minute tolerance
            return False

        return True

    def is_compatible(self, message: DiscoveryAnnounce, my_domain: str, my_schema: str, my_version: str = "1.0.0") -> bool:
        """
        Check if a discovered peer is compatible.

        Compatibility requires matching domain, schema, and major model version.

        Args:
            message: Discovery announce message
            my_domain: This peer's domain
            my_schema: This peer's data schema hash
            my_version: This peer's model version

        Returns:
            True if peers are compatible for gossip
        """
        from quinkgl.topology.base import is_version_compatible
        return (
            message.domain == my_domain and
            message.data_schema_hash == my_schema and
            is_version_compatible(my_version, message.model_version)
        )
