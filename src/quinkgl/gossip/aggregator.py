"""
ModelAggregator

Manages the continuous gossip learning loop including
training, peer selection, model exchange, and aggregation.
"""

import asyncio
import logging
from typing import Any, List, Optional, Callable, Dict
from datetime import datetime

import numpy as np

from quinkgl.gossip.protocol import MessageType, GossipMessage, ModelUpdateMessage, CheckpointAnnounceMessage
from quinkgl.topology.base import TopologyStrategy, SelectionContext, PeerInfo
from quinkgl.aggregation.base import AggregationStrategy, ModelUpdate, AggregatedModel
from quinkgl.models.base import ModelWrapper, TrainingConfig
from quinkgl.observability.events import EventEmitter
from quinkgl.gossip.consensus import ConsensusTracker, PeerCheckpoint
from quinkgl.training.convergence import ConvergenceMonitor, ConvergenceConfig, ConvergenceStatus
from quinkgl.training.quality import compute_peer_similarity, compute_weight_fingerprint

logger = logging.getLogger(__name__)


class ModelAggregator:
    """
    Orchestrates the continuous gossip learning process.

    Manages the training → gossip → aggregation cycle.
    """

    def __init__(
        self,
        peer_id: str,
        domain: str,
        data_schema_hash: str,
        model: ModelWrapper,
        topology: TopologyStrategy,
        aggregator: AggregationStrategy,
        gossip_interval: float = 60.0,
        training_config: Optional[TrainingConfig] = None,
        min_peers_before_aggregate: int = 1,
        checkpoint_interval: int = 10,
        consensus_threshold: float = 0.5,
        consensus_loss_tolerance: float = 0.05,
        convergence_config: Optional[ConvergenceConfig] = None,
        stale_round_tolerance: int = 2,
        min_peers_for_consensus: int = 3,
        max_round_ahead: int = 50
    ):
        """
        Initialize the model aggregator.

        Args:
            peer_id: Unique identifier for this peer
            domain: Domain identifier (e.g., "health", "agriculture")
            data_schema_hash: Hash of data schema for compatibility
            model: Model wrapper for training
            topology: Topology strategy for peer selection
            aggregator: Aggregation strategy for model combining
            gossip_interval: Seconds between gossip rounds
            training_config: Configuration for local training
            min_peers_before_aggregate: Minimum pending updates required before
                aggregation proceeds (default: 1). If fewer updates are
                available, aggregation is deferred to the next round.
            stale_round_tolerance: Maximum allowed round difference for
                incoming updates. Updates with
                ``abs(round_number - current_round) > stale_round_tolerance``
                are silently rejected (default: 2).
            min_peers_for_consensus: Minimum number of peers required
                before consensus can be declared (default: 3).
            max_round_ahead: Maximum allowed round number offset for
                checkpoint recording (default: 50).
        """
        self.peer_id = peer_id
        self.domain = domain
        self.data_schema_hash = data_schema_hash
        self.model = model
        self.topology = topology
        self.aggregator = aggregator
        self.gossip_interval = gossip_interval
        self.training_config = training_config or TrainingConfig()
        self.event_emitter = EventEmitter()
        self.model_version = model.get_model_version() if model else "1.0.0"
        self.min_peers_before_aggregate = min_peers_before_aggregate
        self.consensus_tracker = ConsensusTracker(
            checkpoint_interval=checkpoint_interval,
            consensus_threshold=consensus_threshold,
            loss_tolerance=consensus_loss_tolerance,
            min_peers_for_consensus=min_peers_for_consensus,
            max_round_ahead=max_round_ahead,
        )
        self.convergence_monitor = ConvergenceMonitor(config=convergence_config)

        # State
        self.running = False
        self.current_round = 0
        self.known_peers: Dict[str, PeerInfo] = {}
        self.pending_updates: List[ModelUpdate] = []
        self._pending_lock = asyncio.Lock()
        self._aggregating = False
        self.stale_round_tolerance = stale_round_tolerance
        self._aggregation_event = asyncio.Event()
        self._background_tasks: set[asyncio.Task] = set()
        self._MAX_PENDING_EVENTS = 1024
        self._event_drop_warned = False
        self.metrics: Dict[str, float] = {} # Store latest training metrics
        self.metrics_history: List[Dict] = [] # History for plotting
        self.comm_log: List[Dict] = [] # Log of outgoing messages
        self._last_training_result = None  # Store last TrainingResult for sample_count
        # Task 7b: track per-peer rejection counts to warn on repeated round divergence.
        self._peer_rejection_counts: Dict[str, int] = {}

        # Domain-aware collaboration state (set by GossipNode)
        self._local_fingerprint: Optional[Any] = None
        self._local_fingerprint_provider: Optional[Callable[[int], Any]] = None
        self._local_fingerprint_update_callback: Optional[Callable[[Any], None]] = None
        self._local_manifest_id: Optional[bytes] = None
        self._prototype_store: Optional[Any] = None

        # Network callbacks (to be set by transport layer)
        self.send_message_callback: Optional[Callable] = None
        self.broadcast_callback: Optional[Callable] = None

        # Metrics callback (to be set by transport layer)
        self.metrics_callback: Optional[Callable] = None

        # Lifecycle hooks
        self.hooks = {
            "before_train": [],
            "after_train": [],
            "before_send": [],
            "after_receive": [],
            "before_aggregate": [],
            "after_aggregate": [],
            # MCP monitoring hooks
            "on_training_complete": [],  # Called with TrainingResult
            "on_model_sent": [],         # Called with (peer_ids, model_size)
            "on_aggregation_complete": [],  # Called with AggregatedModel
        }

    def _spawn_task(self, coro) -> Optional[asyncio.Task]:
        """Create a tracked background task that auto-removes on completion."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        task = loop.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _emit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Schedule runtime observability delivery off the hot path.

        Caps pending event tasks at ``_MAX_PENDING_EVENTS`` to prevent
        unbounded growth when subscribers are slow (A3 §2.4).
        """
        if self.event_emitter:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self.event_emitter.emit(event_type, payload)
                return

            # Count pending event-delivery tasks
            event_tasks = sum(
                1 for t in self._background_tasks
                if not t.done() and t.get_name().startswith("evt:")
            )
            if event_tasks >= self._MAX_PENDING_EVENTS:
                if not self._event_drop_warned:
                    logger.warning(
                        f"Event backlog reached {self._MAX_PENDING_EVENTS}, "
                        f"dropping events until subscribers catch up"
                    )
                    self._event_drop_warned = True
                return
            self._event_drop_warned = False

            task = self._spawn_task(self._deliver_event(event_type, payload))
            if task is not None:
                task.set_name(f"evt:{event_type}")

    async def _deliver_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Deliver one observability event on the event loop."""
        self.event_emitter.emit(event_type, payload)

    def _weight_summary(self, weights: Any) -> Dict[str, Any]:
        """Return a compact, deterministic, terminal-safe summary of weights."""
        kind = type(weights).__name__

        if isinstance(weights, np.ndarray):
            return {
                "kind": "ndarray",
                "shape": list(weights.shape),
                "layer_count": 1,
                "total_elements": int(weights.size),
            }

        if isinstance(weights, dict):
            layer_count = 0
            total_elements = 0
            for key, value in sorted(
                weights.items(),
                key=lambda item: (type(item[0]).__name__, repr(item[0])),
            ):
                child = self._weight_summary(value)
                layer_count += int(child.get("layer_count", 0))
                total_elements += int(child.get("total_elements", 0))
            return {
                "kind": "dict",
                "field_count": len(weights),
                "layer_count": layer_count,
                "total_elements": total_elements,
            }

        if isinstance(weights, (list, tuple)):
            layer_count = 0
            total_elements = 0
            for item in weights:
                child = self._weight_summary(item)
                layer_count += int(child.get("layer_count", 0))
                total_elements += int(child.get("total_elements", 0))
            return {
                "kind": kind,
                "item_count": len(weights),
                "layer_count": layer_count,
                "total_elements": total_elements,
            }

        if np.isscalar(weights):
            return {
                "kind": kind,
                "layer_count": 1,
                "total_elements": 1,
            }

        return {
            "kind": kind,
            "layer_count": 1,
            "total_elements": 1,
        }

    def register_hook(self, hook_name: str, callback: Callable):
        """Register a lifecycle hook callback."""
        if hook_name in self.hooks:
            self.hooks[hook_name].append(callback)
        else:
            raise ValueError(f"Unknown hook: {hook_name}")

    async def _execute_hooks(self, hook_name: str, *args, **kwargs):
        """Execute all callbacks for a hook.

        Task 2a: each callback is wrapped in a try/except so a single failing
        hook cannot abort the current pipeline step.
        """
        for callback in self.hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Hook '{hook_name}' raised an exception and was skipped: "
                    f"{e.__class__.__name__}: {e}"
                )

    def add_peer(self, peer_info: PeerInfo):
        """Add a newly discovered peer."""
        if peer_info.peer_id not in self.known_peers:
            logger.debug(f"Discovered new peer: {peer_info.peer_id}")
            self.known_peers[peer_info.peer_id] = peer_info
            self._emit_event(
                "peer_discovered",
                {
                    "node_id": self.peer_id,
                    "peer_id": peer_info.peer_id,
                    "domain": peer_info.domain,
                    "data_schema_hash": peer_info.data_schema_hash,
                    "round": self.current_round,
                },
            )

            # Notify topology strategy with error handling
            async def _notify_topology():
                try:
                    await self.topology.on_new_peer_discovered(peer_info)
                except Exception as e:
                    logger.error(f"Error notifying topology about new peer {peer_info.peer_id}: {e}")

            self._spawn_task(_notify_topology())

    async def remove_peer(self, peer_id: str):
        """Remove a disconnected peer."""
        if peer_id in self.known_peers:
            logger.debug(f"Removing peer: {peer_id}")
            del self.known_peers[peer_id]
            self._emit_event(
                "peer_disconnected",
                {
                    "node_id": self.peer_id,
                    "peer_id": peer_id,
                    "round": self.current_round,
                },
            )
            
            # Notify topology strategy
            await self.topology.on_peer_disconnected(peer_id)
            logger.debug(f"Removed peer: {peer_id}")

    async def handle_incoming_message(self, message: GossipMessage):
        """
        Handle an incoming message from a peer.

        Args:
            message: The received message
        """
        if message.msg_type == MessageType.MODEL_UPDATE:
            await self._handle_model_update(message)
        elif message.msg_type == MessageType.HEARTBEAT:
            if message.sender_id in self.known_peers:
                self.known_peers[message.sender_id].last_seen = datetime.now()
        elif message.msg_type == MessageType.DISCOVERY_ANNOUNCE:
            await self._handle_discovery_announce(message)
        elif message.msg_type == MessageType.CHECKPOINT_ANNOUNCE:
            await self._handle_checkpoint_announce(message)

    async def _handle_model_update(self, message: ModelUpdateMessage):
        """Handle an incoming model update.

        Rejects updates when the loop is not running (A4) and applies
        round-gating so that stale or implausibly future updates never
        enter ``pending_updates`` (A1 §2.2).
        """
        # A4: refuse appends once the loop has stopped
        if not self.running:
            logger.debug(
                f"Dropping update from {message.sender_id}: loop not running"
            )
            return

        await self._execute_hooks("after_receive", message)

        # Round-gate: reject updates too far from current round
        round_diff = abs(message.round_number - self.current_round)
        if round_diff > self.stale_round_tolerance:
            # Task 7b: track rejections per peer and warn when a peer is repeatedly
            # rejected — a sign that it may be permanently isolated.
            count = self._peer_rejection_counts.get(message.sender_id, 0) + 1
            self._peer_rejection_counts[message.sender_id] = count
            if count == 1 or count % 5 == 0:
                logger.warning(
                    f"Peer {message.sender_id} rejected {count} time(s) due to round "
                    f"divergence (msg_round={message.round_number}, "
                    f"local_round={self.current_round}, "
                    f"tolerance={self.stale_round_tolerance}). "
                    "Peer may be permanently isolated — consider increasing "
                    "stale_round_tolerance or investigating network partitioning."
                )
            else:
                logger.debug(
                    f"Rejecting stale/future update from {message.sender_id}: "
                    f"msg_round={message.round_number}, local_round={self.current_round}, "
                    f"tolerance={self.stale_round_tolerance}"
                )
            self._emit_event(
                "model_rejected_stale",
                {
                    "node_id": self.peer_id,
                    "peer_id": message.sender_id,
                    "msg_round": message.round_number,
                    "local_round": self.current_round,
                    "tolerance": self.stale_round_tolerance,
                },
            )
            return

        # Create ModelUpdate from message
        update = ModelUpdate(
            peer_id=message.sender_id,
            weights=message.weights,
            sample_count=message.sample_count,
            loss=message.loss,
            accuracy=message.accuracy,
            round_number=message.round_number
        )

        async with self._pending_lock:
            self.pending_updates.append(update)

        self._emit_event(
            "model_received",
            {
                "node_id": self.peer_id,
                "round": message.round_number,
                "peer_id": message.sender_id,
                "sample_count": message.sample_count,
                "weight_summary": self._weight_summary(message.weights),
                "loss": message.loss,
                "accuracy": message.accuracy,
            },
        )
        logger.debug(f"Received model update from {message.sender_id}")

        # Trigger aggregation event to notify the main loop
        self._aggregation_event.set()

    async def _handle_discovery_announce(self, message: GossipMessage):
        """Handle a discovery announcement."""
        if (message.payload.get("domain") == self.domain and
            message.payload.get("data_schema_hash") == self.data_schema_hash):
            peer_info = PeerInfo(
                peer_id=message.sender_id,
                domain=message.payload["domain"],
                data_schema_hash=message.payload["data_schema_hash"],
                model_version=message.payload.get("model_version", "1.0.0")
            )
            self.add_peer(peer_info)

    async def _handle_checkpoint_announce(self, message):
        """Handle a checkpoint announcement from a peer."""
        checkpoint = PeerCheckpoint(
            peer_id=message.sender_id,
            round_number=message.round_number,
            loss=message.loss,
            accuracy=message.accuracy,
            model_version=message.model_version,
        )
        self.consensus_tracker.record_checkpoint(checkpoint)

        result = self.consensus_tracker.check_consensus()
        if result and result.reached:
            self._emit_event(
                "consensus_reached",
                {
                    "node_id": self.peer_id,
                    "round": result.checkpoint_round,
                    "agreeing_peers": result.agreeing_peers,
                    "total_peers": result.total_peers,
                    "agreement_ratio": result.agreement_ratio,
                    "mean_loss": result.mean_loss,
                    "mean_accuracy": result.mean_accuracy,
                },
            )
            logger.debug(
                f"Consensus reached at round {result.checkpoint_round}: "
                f"{result.agreeing_peers}/{result.total_peers} peers agree "
                f"(ratio={result.agreement_ratio:.2f})"
            )

    async def _broadcast_checkpoint(self, loss: float = 0.0, accuracy: float = 0.0) -> None:
        """Broadcast checkpoint announcement to all known peers."""
        self.consensus_tracker.record_checkpoint(
            PeerCheckpoint(
                peer_id=self.peer_id,
                round_number=self.current_round,
                loss=loss,
                accuracy=accuracy,
                model_version=self.model_version,
            )
        )

        if self.broadcast_callback:
            checkpoint_msg = CheckpointAnnounceMessage.create(
                sender_id=self.peer_id,
                round_number=self.current_round,
                loss=loss,
                accuracy=accuracy,
                model_version=self.model_version,
            )
            try:
                await self.broadcast_callback(checkpoint_msg)
            except Exception as e:
                logger.debug(f"Failed to broadcast checkpoint: {e}")

    async def _train_local(self, data) -> tuple:
        """Perform local training. Returns (loss, accuracy, samples_trained) tuple."""
        await self._execute_hooks("before_train")
        self._emit_event(
            "training_started",
            {
                "node_id": self.peer_id,
                "round": self.current_round,
                "loss": None,
                "accuracy": None,
                "samples_trained": 0,
            },
        )

        # Inject FedProx proximal term into training config if applicable
        training_config = self.training_config
        if hasattr(self.aggregator, 'get_training_config_overrides'):
            overrides = self.aggregator.get_training_config_overrides()
            if overrides:
                from quinkgl.models.base import TrainingConfig
                training_config = TrainingConfig(
                    epochs=self.training_config.epochs,
                    batch_size=self.training_config.batch_size,
                    learning_rate=self.training_config.learning_rate,
                    verbose=self.training_config.verbose,
                    on_epoch_end=self.training_config.on_epoch_end,
                    loss_fn=self.training_config.loss_fn,
                    optimizer=self.training_config.optimizer,
                    optimizer_kwargs=self.training_config.optimizer_kwargs,
                    grad_clip_norm=self.training_config.grad_clip_norm,
                    proximal_coefficient=overrides.get("proximal_coefficient"),
                    global_weights=overrides.get("global_weights"),
                )

        result = await self.model.train(data, training_config)

        # Store result for sample_count in aggregation
        self._last_training_result = result

        await self._execute_hooks("after_train", result)

        loss = result.final_loss if result.final_loss is not None else 0.0
        acc = result.final_accuracy if result.final_accuracy is not None else 0.0
        samples = result.samples_trained if result.samples_trained > 0 else self.training_config.batch_size

        acc_str = f"{acc:.4f}" if result.final_accuracy is not None else "N/A"
        logger.debug(
            f"Local training round {self.current_round} complete: "
            f"loss={loss:.4f}, acc={acc_str}, samples={samples}"
        )

        # Update metrics if callback is registered
        if self.metrics_callback:
            self.metrics_callback(loss=loss, accuracy=acc, round_num=self.current_round)

        # MCP hook: training complete
        await self._execute_hooks("on_training_complete", result)
        self._emit_event(
            "training_completed",
            {
                "node_id": self.peer_id,
                "round": self.current_round,
                "loss": loss,
                "accuracy": acc,
                "samples_trained": samples,
            },
        )

        return loss, acc, samples

    async def _send_model(self, target_peers: List[str], loss: float = None, accuracy: float = None, samples_trained: int = None) -> None:
        """Send current model to target peers concurrently.

        Each peer send is wrapped in its own ``try/except`` so that a
        single flaky peer cannot break the round or bump
        ``consecutive_errors`` (A2 §2.5).  All targets are dispatched
        via ``asyncio.gather`` for parallelism.
        """
        from quinkgl.models.base import PersonalizedModelWrapper

        if isinstance(self.model, PersonalizedModelWrapper):
            weights = self.model.get_backbone_weights()
        else:
            weights = self.model.get_weights()

        await self._execute_hooks("before_send", weights)

        sample_count = samples_trained if samples_trained is not None else self.training_config.batch_size

        model_message = ModelUpdateMessage.create(
            sender_id=self.peer_id,
            weights=weights,
            sample_count=sample_count,
            round_number=self.current_round,
            loss=loss,
            accuracy=accuracy
        )

        self._emit_event(
            "model_send_started",
            {
                "node_id": self.peer_id,
                "round": self.current_round,
                "peer_ids": list(target_peers),
                "sample_count": sample_count,
                "weight_summary": self._weight_summary(weights),
                "loss": loss,
                "accuracy": accuracy,
            },
        )

        if not self.send_message_callback:
            logger.debug("send_message_callback is None, skipping send")
            return

        sent_peers: List[str] = []
        failed_peers: List[str] = []

        async def _send_to_peer(peer_id: str):
            try:
                await self.send_message_callback(peer_id, model_message)
                sent_peers.append(peer_id)
                logger.debug(f"Sent model update to {peer_id}")
                self.comm_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "target": peer_id,
                    "round": self.current_round,
                })
                if len(self.comm_log) > 50:
                    self.comm_log.pop(0)
            except Exception as e:
                failed_peers.append(peer_id)
                logger.warning(
                    f"Failed to send model to {peer_id}: "
                    f"{e.__class__.__name__}: {e}"
                )
                self.comm_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "target": peer_id,
                    "round": self.current_round,
                    "error": str(e),
                })
                if len(self.comm_log) > 50:
                    self.comm_log.pop(0)

        await asyncio.gather(*(_send_to_peer(pid) for pid in target_peers))

        if failed_peers:
            self._emit_event(
                "model_send_failed",
                {
                    "node_id": self.peer_id,
                    "round": self.current_round,
                    "failed_peers": failed_peers,
                },
            )

        if sent_peers:
            import sys
            model_size = sys.getsizeof(weights) if weights else 0
            await self._execute_hooks("on_model_sent", sent_peers, model_size)
            self._emit_event(
                "model_sent",
                {
                    "node_id": self.peer_id,
                    "round": self.current_round,
                    "peer_ids": list(sent_peers),
                    "sample_count": sample_count,
                    "weight_summary": self._weight_summary(weights),
                    "loss": loss,
                    "accuracy": accuracy,
                },
            )

    async def _aggregate_models(self) -> Optional[AggregatedModel]:
        """Aggregate pending model updates.

        Uses an async lock to atomically drain ``pending_updates`` into a
        local batch so that updates arriving during the (potentially slow)
        aggregation call are preserved for the next round instead of being
        silently discarded (A1 §2.1 TOCTOU fix).

        A re-entrancy guard (``_aggregating``) prevents overlapping
        aggregation calls (A1 §2.6).
        """
        # Re-entrancy guard
        if self._aggregating:
            logger.debug("Aggregation already in progress, skipping")
            return None

        # ── Drain under lock ────────────────────────────────────────
        async with self._pending_lock:
            if not self.pending_updates:
                return None
            if len(self.pending_updates) < self.min_peers_before_aggregate:
                logger.debug(
                    f"Deferring aggregation: {len(self.pending_updates)} pending "
                    f"< min_peers_before_aggregate={self.min_peers_before_aggregate}"
                )
                # Task 12a: do NOT clear _aggregation_event here. Clearing it would
                # force the next round to sleep the full gossip_interval even though
                # there are queued updates waiting for more peers to arrive.
                return None
            # Drain into local batch; the shared list is now empty so new
            # updates arriving during aggregation are safely appended.
            batch = list(self.pending_updates)
            self.pending_updates.clear()

        # Belt-and-braces: filter the batch against current_round one more
        # time, in case round incremented between append and drain.
        batch = [
            u for u in batch
            if abs(u.round_number - self.current_round) <= self.stale_round_tolerance
        ]
        if not batch:
            return None

        self._aggregating = True
        try:
            await self._execute_hooks("before_aggregate", batch)

            # Include own model in aggregation
            if self._last_training_result and self._last_training_result.samples_trained > 0:
                own_sample_count = self._last_training_result.samples_trained
            else:
                peer_counts = [u.sample_count for u in batch if u.sample_count > 0]
                own_sample_count = (
                    sum(peer_counts) // len(peer_counts)
                    if peer_counts
                    else self.training_config.batch_size
                )

            from quinkgl.models.base import PersonalizedModelWrapper as _PMW
            if isinstance(self.model, _PMW):
                own_weights = self.model.get_backbone_weights()
            else:
                own_weights = self.model.get_weights()

            own_update = ModelUpdate(
                peer_id=self.peer_id,
                weights=own_weights,
                sample_count=own_sample_count,
                round_number=self.current_round,
            )

            all_updates = [own_update] + batch

            try:
                aggregated = await self.aggregator.aggregate(all_updates)
            except Exception:
                # Task 1a+1b: re-insert the drained batch under lock so updates
                # are not silently lost when aggregation fails.
                async with self._pending_lock:
                    self.pending_updates[:0] = batch
                raise

            await self._execute_hooks("after_aggregate", aggregated)

            # Update model with aggregated weights
            from quinkgl.models.base import PersonalizedModelWrapper, APFLMixin

            if isinstance(self.model, PersonalizedModelWrapper):
                self.model.set_backbone_weights(aggregated.weights)
            else:
                self.model.set_weights(aggregated.weights)

            # APFL adaptive mixing: blend local + global weights
            if isinstance(self.model, PersonalizedModelWrapper) and isinstance(self.model, APFLMixin):
                local_weights = self.model.get_head_weights()
                global_weights = aggregated.weights
                mixed = self.model.compute_personalized_weights(
                    local_weights=local_weights,
                    global_weights=global_weights,
                )
                self.model.set_backbone_weights(mixed)

            self._aggregation_event.clear()

            # S11a: reset error-feedback residuals after aggregation so the buffer
            # is not stale against the new (aggregated) weights.
            from quinkgl.serialization.error_feedback import ErrorFeedbackState as _EFS
            if hasattr(self.model, '_ef_state') and isinstance(self.model._ef_state, _EFS):
                self.model._ef_state.reset()
        finally:
            self._aggregating = False

        logger.debug(
            f"Aggregated models from {len(aggregated.contributing_peers)} peers "
            f"(total_samples={aggregated.total_samples})"
        )

        # MCP hook: aggregation complete
        await self._execute_hooks("on_aggregation_complete", aggregated)

        # Quality assessment: compute peer similarity
        peer_weights = [u.weights for u in all_updates if u.weights is not None]
        if len(peer_weights) >= 2:
            similarity = compute_peer_similarity(peer_weights)
            aggregated.metadata["peer_similarity"] = similarity
            if similarity.get("mean_similarity", 0) > 0.95:
                self._emit_event(
                    "models_converged",
                    {
                        "node_id": self.peer_id,
                        "round": self.current_round,
                        "mean_similarity": similarity["mean_similarity"],
                        "peer_count": similarity["peer_count"],
                    },
                )

        # Weight fingerprint for lightweight comparison
        aggregated.metadata["weight_fingerprint"] = compute_weight_fingerprint(aggregated.weights)

        self._emit_event(
            "aggregation_completed",
            {
                "node_id": self.peer_id,
                "round": self.current_round,
                "peer_ids": list(aggregated.contributing_peers),
                "sample_count": aggregated.total_samples,
                "weight_summary": self._weight_summary(aggregated.weights),
            },
        )

        return aggregated

    def _set_local_fingerprint(self, fingerprint: Optional[Any]) -> None:
        self._local_fingerprint = fingerprint
        if self._local_fingerprint_update_callback is not None:
            self._local_fingerprint_update_callback(fingerprint)

    def _refresh_local_fingerprint(self) -> None:
        if self._local_fingerprint_provider is None:
            return
        fingerprint = self._local_fingerprint_provider(self.current_round)
        self._set_local_fingerprint(fingerprint)

    async def run_continuous(self, data_provider=None, eval_data_provider=None):
        """
        Run the continuous gossip learning loop.

        Args:
            data_provider: Callable (or dataset) for local training each round.
            eval_data_provider: Optional callable (or dataset) for post-aggregation
                evaluation.  When provided, the model is evaluated on this data
                **after** aggregation and the resulting metrics are used for the
                checkpoint broadcast instead of the pre-aggregation training metrics
                (Task 6a).  Pass a small validation split to keep evaluation cheap.
        """
        self.running = True
        logger.info("Starting continuous gossip learning loop")

        # Task 3a: warn early when no training data is provided so the operator
        # is not surprised by a node that gossips untrained weights indefinitely.
        if data_provider is None:
            logger.warning(
                f"Node {self.peer_id}: run_continuous() called without data or "
                "data_provider. Training will be skipped and untrained weights "
                "will be gossiped each round."
            )

        consecutive_errors = 0
        max_consecutive_errors = 5

        try:
            while self.running:
                round_start_time = datetime.now()

                try:
                    # Task 5a: reset _last_training_result at the top of each round
                    # so a stale result from a previous successful round is never used
                    # for own_sample_count when this round's training fails.
                    self._last_training_result = None

                    # Task 4a: increment AFTER all work succeeds so the counter
                    # reflects completed rounds, not attempted ones.
                    # NOTE: all in-round references to self.current_round use the
                    # previous round's number; peers' round-gating tolerance handles
                    # the 1-round offset (stale_round_tolerance ≥ 1).
                    self.current_round += 1
                    self._refresh_local_fingerprint()

                    loss, acc, samples = 0.0, 0.0, 0
                    trained_this_round = False

                    # 1. Local training
                    if data_provider:
                        train_data = data_provider() if callable(data_provider) else data_provider
                        loss, acc, samples = await self._train_local(train_data)
                        trained_this_round = True

                        # Apply EMA smoothing (alpha=0.2) to reduce jitter from small batches
                        alpha = 0.2
                        if not self.metrics:
                            self.metrics = {"loss": loss, "accuracy": acc}
                        else:
                            self.metrics = {
                                "loss": alpha * loss + (1 - alpha) * self.metrics.get("loss", loss),
                                "accuracy": alpha * acc + (1 - alpha) * self.metrics.get("accuracy", acc)
                            }

                        # Log Metrics History
                        self.metrics_history.append({
                            "round": self.current_round,
                            "loss": self.metrics["loss"],
                            "accuracy": self.metrics["accuracy"],
                            "timestamp": datetime.now().isoformat()
                        })
                        if len(self.metrics_history) > 100:
                            self.metrics_history.pop(0)

                        # Task 10a: pass raw loss/acc (not EMA-smoothed self.metrics)
                        # so the convergence monitor's own sliding window is not
                        # double-smoothed, which would delay early stopping.
                        convergence_report = self.convergence_monitor.update(
                            loss=loss,
                            accuracy=acc,
                            round_number=self.current_round,
                        )
                        if self.convergence_monitor.should_stop(convergence_report):
                            self._emit_event(
                                "early_stopping",
                                {
                                    "node_id": self.peer_id,
                                    "round": self.current_round,
                                    "status": convergence_report.status.value,
                                    "best_loss": convergence_report.best_loss,
                                    "best_accuracy": convergence_report.best_accuracy,
                                    "rounds_without_improvement": convergence_report.rounds_without_improvement,
                                },
                            )
                            logger.debug(
                                f"Early stopping triggered at round {self.current_round}: "
                                f"status={convergence_report.status.value}, "
                                f"best_loss={convergence_report.best_loss:.4f}, "
                                f"rounds_without_improvement={convergence_report.rounds_without_improvement}"
                            )
                            self.running = False
                            break

                    # 2. Select gossip targets
                    my_fingerprint = None
                    if hasattr(self, '_local_fingerprint') and self._local_fingerprint is not None:
                        my_fingerprint = self._local_fingerprint

                    my_manifest_id = None
                    if hasattr(self, '_local_manifest_id') and self._local_manifest_id is not None:
                        my_manifest_id = self._local_manifest_id

                    context = SelectionContext(
                        my_peer_id=self.peer_id,
                        my_domain=self.domain,
                        my_data_schema_hash=self.data_schema_hash,
                        known_peers=list(self.known_peers.values()),
                        current_round=self.current_round,
                        my_model_version=self.model_version,
                        my_fingerprint=my_fingerprint,
                        my_manifest_id=my_manifest_id,
                    )
                    candidate_count = len(context.get_compatible_peers(exclude_self=True))
                    targets = await self.topology.select_targets(context, count=3)
                    self._emit_event(
                        "targets_selected",
                        {
                            "node_id": self.peer_id,
                            "round": self.current_round,
                            "candidate_count": candidate_count,
                            "selected_targets": list(targets),
                        },
                    )

                    # 3. Send model to targets (with metrics)
                    # Task 3b: skip sending when no training has ever occurred to
                    # avoid gossiping a fully untrained model.
                    if targets:
                        if not trained_this_round and self._last_training_result is None:
                            logger.debug(
                                "Skipping model send: no training has occurred this "
                                "round and no prior training result is available."
                            )
                        else:
                            await self._send_model(targets, loss=loss, accuracy=acc, samples_trained=samples)

                    # 4. Topology Maintenance (e.g. Shuffle)
                    await self.topology.periodic_maintenance(context)

                    # 5. Wait for incoming models & aggregation trigger
                    # We wait for the gossip interval, but allow interruption for earlier aggregation
                    try:
                        await asyncio.wait_for(self._aggregation_event.wait(), timeout=self.gossip_interval)
                    except asyncio.TimeoutError:
                        pass  # Timeout is normal, just proceed to next round

                    # 6. Aggregate received models
                    await self._aggregate_models()

                    # 7. Checkpoint & consensus
                    # Task 6a: if an eval_data_provider is supplied, evaluate the model
                    # on the validation set after aggregation so the checkpoint reflects
                    # the post-aggregation model quality, not the pre-aggregation training
                    # metrics.  Evaluation is run only on checkpoint rounds to limit cost.
                    checkpoint_loss, checkpoint_acc = loss, acc
                    if self.consensus_tracker.should_checkpoint(self.current_round):
                        if eval_data_provider is not None:
                            try:
                                eval_data = (
                                    eval_data_provider()
                                    if callable(eval_data_provider)
                                    else eval_data_provider
                                )
                                # evaluate() is synchronous; run in executor so we don't
                                # block the event loop during GPU/CPU-bound inference.
                                loop = asyncio.get_running_loop()
                                eval_metrics = await loop.run_in_executor(
                                    None, lambda: self.model.evaluate(eval_data)
                                )
                                checkpoint_loss = float(eval_metrics.get("loss", loss))
                                checkpoint_acc = float(eval_metrics.get("accuracy", acc))
                                logger.debug(
                                    f"Post-aggregation eval round {self.current_round}: "
                                    f"loss={checkpoint_loss:.4f}, acc={checkpoint_acc:.4f}"
                                )
                                self._emit_event(
                                    "post_aggregation_eval",
                                    {
                                        "node_id": self.peer_id,
                                        "round": self.current_round,
                                        "loss": checkpoint_loss,
                                        "accuracy": checkpoint_acc,
                                    },
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Post-aggregation evaluation failed, falling back to "
                                    f"training metrics: {e.__class__.__name__}: {e}"
                                )
                        await self._broadcast_checkpoint(checkpoint_loss, checkpoint_acc)
                        result = self.consensus_tracker.check_consensus()
                        if result and result.reached:
                            self._emit_event(
                                "consensus_reached",
                                {
                                    "node_id": self.peer_id,
                                    "round": result.checkpoint_round,
                                    "agreeing_peers": result.agreeing_peers,
                                    "total_peers": result.total_peers,
                                    "agreement_ratio": result.agreement_ratio,
                                    "mean_loss": result.mean_loss,
                                    "mean_accuracy": result.mean_accuracy,
                                },
                            )
                            logger.debug(
                                f"Consensus reached at round {result.checkpoint_round}: "
                                f"{result.agreeing_peers}/{result.total_peers} peers agree"
                            )
                        self.consensus_tracker.prune_old_checkpoints()

                    # Reset error counter on successful round
                    consecutive_errors = 0

                    round_duration = (datetime.now() - round_start_time).total_seconds()
                    logger.debug(f"Round {self.current_round} completed in {round_duration:.2f}s")

                except asyncio.CancelledError:
                    logger.info("Gossip loop cancelled")
                    break
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(
                        f"Error in round {self.current_round}: {e.__class__.__name__}: {e}"
                    )

                    # Check if we've had too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(
                            f"Too many consecutive errors ({consecutive_errors}). "
                            f"Stopping gossip loop."
                        )
                        raise RuntimeError(
                            f"Gossip loop stopped after {consecutive_errors} consecutive errors"
                        ) from e

                    # Clear aggregation event and continue to next round
                    self._aggregation_event.clear()

                    # Wait a bit before retrying
                    await asyncio.sleep(min(2 ** consecutive_errors, 30))

        finally:
            self.running = False
            # Task 8a+8b: run one final aggregation pass so pending updates are
            # not discarded on graceful shutdown (early-stopping, stop() call, etc.).
            async with self._pending_lock:
                pending_count = len(self.pending_updates)
            if pending_count >= self.min_peers_before_aggregate:
                try:
                    await self._aggregate_models()
                except Exception as e:
                    logger.warning(f"Final aggregation on shutdown failed: {e}")
            async with self._pending_lock:
                self.pending_updates.clear()
            await self._cancel_background_tasks()
            logger.info(f"Gossip learning loop stopped (completed {self.current_round} rounds)")

    def increment_round(self):
        """Manually increment the current round number."""
        self.current_round += 1
        logger.debug(f"Round incremented to {self.current_round}")

    async def _cancel_background_tasks(self):
        """Cancel and await all tracked background tasks."""
        tasks = list(self._background_tasks)
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()

    def stop(self):
        """Stop the gossip learning loop."""
        self.running = False
        logger.info("Stopping continuous gossip learning loop")


# Backward compatibility alias (deprecated - use ModelAggregator instead)
GossipOrchestrator = ModelAggregator
