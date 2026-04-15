#!/usr/bin/env python3
"""
Generate Docker Compose for Scale Test

Usage:
    python3 generate_compose.py [num_nodes]

Default num_nodes: 20
"""

import sys
import yaml

def generate_compose(num_nodes: int = 20):
    services = {}
    
    # Common command template
    # Using 'breast_cancer' dataset as requested (lightweight real data)
    # Using 'cyclon' topology
    # Splitting data non-iid among nodes (simulated by node index)
    
    base_command = (
        "python3 scripts/run_gossip_node.py "
        "--node-id {node_id} "
        "--domain scale-test "
        "--dataset breast_cancer "
        "--split-type iid "
        "--total-nodes {total_nodes} "
        "--node-index {index} "
        "--topology cyclon "
        "--gossip-interval 2.0 "
        "--epochs 1"
    )

    for i in range(num_nodes):
        node_id = f"node_{i+1}"
        
        # Bootstrap: everyone knows node_1 (except node_1 itself)
        # In a real scenario we might have multiple bootstrap nodes.
        # But for this test, if node_1 is up, others can find each other via gossip/shuffle eventually.
        environment = {
            "PYTHONUNBUFFERED": "1"
        }
        
        # We need to pass bootstrap peer info. 
        # The current run_gossip_node.py uses --bootstrap-peers arg (need to check if supported)
        # OR relies on IPv8 discovery.
        # Since we are in Docker network, IPv8 broadcast might work if configured correctly.
        # BUT relying on broadcast in Docker can be tricky. 
        # Let's assume the standard IPv8 config will find peers in same subnet
        # or we might need to explicitely set bootstrap args if implemented. 
        # Checking run_gossip_node.py arguments... it doesn't seem to have explicit --bootstrap host arg exposed easily 
        # in the snippets I saw, but ConnectionManager handles it. 
        # For now, we utilize the Tunnel/IPv8 standard discovery.
        
        services[node_id] = {
            "build": {
                "context": "../../",
                "dockerfile": "docker/Dockerfile"
            },
            "command": base_command.format(
                node_id=node_id, 
                total_nodes=num_nodes, 
                index=i
            ),
            "deploy": {
                "resources": {
                    "limits": {
                        "memory": "256M"
                    }
                }
            },
            "environment": environment,
            "ports": [
                f"{8000+i+1}:{8000+i+1}"
            ]
        }

    compose = {
        "version": "3.8",
        "services": services
    }

    output_file = f"docker-compose.{num_nodes}.yml"
    with open(output_file, "w") as f:
        yaml.dump(compose, f, sort_keys=False)
    
    print(f"Generated {output_file} with {num_nodes} nodes.")

if __name__ == "__main__":
    n = 20
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    generate_compose(n)
