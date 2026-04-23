"""
IPv8 Manager for QuinkGL P2P Networking

Manages IPv8 instance lifecycle and community registration.
"""

import asyncio
import logging
from typing import List, Optional

from ipv8.configuration import ConfigBuilder, Strategy, WalkerDefinition, default_bootstrap_defs
from ipv8.community import Community
from ipv8_service import IPv8

logger = logging.getLogger(__name__)


class IPv8Manager:
    """
    Manages IPv8 instance and QuinkGL community.
    
    Handles:
    - IPv8 instance creation and configuration
    - Community registration
    - Peer discovery
    - Lifecycle management
    """
    
    def __init__(self, node_id: str, port: int = 0):
        """
        Initialize IPv8 Manager.
        
        Args:
            node_id: Unique identifier for this node
            port: UDP port (0 for random)
        """
        self.node_id = node_id
        self.port = port
        self.ipv8: Optional[IPv8] = None
        self.community: Optional[Community] = None
        self.running = False
        self.config = None # Added to store the finalized config
        
        logger.debug(f"IPv8Manager initialized for node '{node_id}'")
    
    async def start(self, community_class=None, node_id_param=None):
        """
        Start IPv8 instance with community.

        Args:
            community_class: Community class to load
            node_id_param: Node ID to pass to community
        """
        if self.running:
            logger.warning("IPv8 already running")
            return

        try:
            # Create working directory if it doesn't exist (portable for Windows)
            import os
            import tempfile
            temp_dir = tempfile.gettempdir()
            work_dir = os.path.join(temp_dir, f"ipv8_quinkgl_{self.node_id}")
            key_file = os.path.join(temp_dir, f"ipv8_quinkgl_{self.node_id}.pem")
            os.makedirs(work_dir, exist_ok=True)
            if not getattr(self, 'last_seen_round_state_path', ''):
                self.last_seen_round_state_path = os.path.join(work_dir, 'last_seen_round.json')

            # Build IPv8 configuration
            builder = ConfigBuilder().clear_keys().clear_overlays()

            # Add key for this peer
            builder.add_key("my peer", "medium", key_file)

            # Set port configuration
            builder.set_port(self.port)

            # Enable NAT traversal
            builder.set_address("0.0.0.0")

            # Set working directory
            builder.set_working_directory(work_dir)
            
            # Add overlay if community class provided
            if community_class:
                from ipv8.configuration import Strategy, WalkerDefinition, default_bootstrap_defs

                # Use Tribler's public bootstrap nodes for peer discovery
                bootstrap_nodes = default_bootstrap_defs

                # Build settings dict with community parameters
                # These will be passed as kwargs to community __init__
                community_settings = {}
                if hasattr(self, 'domain'):
                    community_settings['domain'] = self.domain
                if hasattr(self, 'data_schema_hash'):
                    community_settings['data_schema_hash'] = self.data_schema_hash
                if hasattr(self, 'require_signature'):
                    community_settings['require_signature'] = self.require_signature
                if hasattr(self, 'last_seen_round_state_path'):
                    community_settings['last_seen_round_state_path'] = self.last_seen_round_state_path
                if hasattr(self, 'max_round_skip'):
                    community_settings['max_round_skip'] = self.max_round_skip

                builder.add_overlay(
                    community_class.__name__,
                    "my peer",
                    [WalkerDefinition(Strategy.RandomWalk, 10, {"timeout": 3.0})],
                    bootstrap_nodes,  # Use public bootstrap nodes
                    {**community_settings, "node_id": node_id_param or self.node_id},  # Initialize args + settings
                    []   # No on_start
                )
            
            # Add DHT Discovery Community for peer discovery
            # This allows peers to find each other globally via DHT
            builder.add_overlay(
                "DHTDiscoveryCommunity",
                "my peer",
                [WalkerDefinition(Strategy.RandomWalk, 20, {"timeout": 3.0})],
                default_bootstrap_defs,
                {},
                []
            )
            
            # Build and start IPv8
            self.ipv8 = IPv8(
                builder.finalize(),
                extra_communities={
                    community_class.__name__: community_class,
                    "DHTDiscoveryCommunity": __import__('ipv8.dht.discovery', fromlist=['DHTDiscoveryCommunity']).DHTDiscoveryCommunity
                } if community_class else {}
            )
            await self.ipv8.start()
            
            # Get reference to community
            if community_class:
                for overlay in self.ipv8.overlays:
                    if isinstance(overlay, community_class):
                        self.community = overlay
                        # CRITICAL: Set the correct node_id (was 'unknown')
                        self.community.node_id = node_id_param or self.node_id
                        logger.debug(f"Set community node_id to: {self.community.node_id}")
                        
                        # CRITICAL: Manually start the community (IPv8 doesn't auto-start)
                        if not hasattr(self.community, '_started'):
                            self.community.started()
                            logger.debug(f"Manually started community")
                        
                        break
                
                if not self.community:
                    raise RuntimeError(f"Failed to find {community_class.__name__} in overlays")
            
            self.running = True
            logger.debug(f"IPv8 started on port {self.ipv8.endpoint.get_address()[1]}")
            
        except Exception as e:
            logger.error(f"Failed to start IPv8: {e}", exc_info=True)
            raise
    
    async def register_community(self, community_class, *args, **kwargs):
        """
        Register a community overlay.
        
        Note: This method is deprecated. Use start() with community_class parameter instead.
        """
        logger.warning("register_community() is deprecated, use start(community_class) instead")
        await self.start(community_class, kwargs.get('node_id'))
    
    def get_peers(self) -> List:
        """
        Get list of discovered peers.
        
        Returns:
            List of Peer objects
        """
        if not self.community:
            return []
        
        return self.community.get_peers()
    
    def find_peer_by_id(self, peer_id: str):
        """
        Find peer by node ID.
        
        Args:
            peer_id: Node ID to search for
            
        Returns:
            Peer object or None
        """
        # Note: This requires custom peer tracking in community
        # Will be implemented in QuinkGLCommunity
        if not self.community:
            return None
        
        # Placeholder - actual implementation in community
        for peer in self.community.get_peers():
            # Custom attribute check (to be added in community)
            if hasattr(peer, 'node_id') and peer.node_id == peer_id:
                return peer
        
        return None
    
    async def stop(self):
        """Stop IPv8 gracefully.

        B8: Idempotent — safe to call even if start() never succeeded or
        was only partially completed.
        """
        try:
            if self.ipv8:
                await self.ipv8.stop()
                self.ipv8 = None
        except Exception as e:
            logger.error(f"Error stopping IPv8: {e}")
        finally:
            self.running = False
            self.community = None
            logger.debug("IPv8 stopped")
    
    def get_stats(self) -> dict:
        """
        Get IPv8 statistics.
        
        Returns:
            Dictionary with stats
        """
        if not self.ipv8 or not self.community:
            return {
                "running": False,
                "peers": 0,
                "port": self.port
            }
        
        return {
            "running": self.running,
            "peers": len(self.get_peers()),
            "port": self.ipv8.endpoint.get_address()[1],
            "community_id": self.community.community_id.hex() if self.community else None
        }
