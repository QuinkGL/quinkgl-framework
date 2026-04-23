# QuinkGL Simulation Tests

This directory contains simulation scripts for testing QuinkGL in various scenarios.

## Scripts

### `script.py` (or `script-1.py`)
Single-node experiment script for QuinkGL.

This file is intentionally verbose and comment-heavy. The goal is not to be the shortest possible script, but to be the easiest place to edit while you are experimenting with:
- Topology strategies
- Aggregation strategies
- Training hyperparameters
- Synthetic data size / shape
- Node identity / domain / port

**How to use:**
1. Edit the CONFIG section in the script.
2. Run in one terminal: `python script.py`
3. For multi-node gossip, run in multiple terminals with different NODE_ID and PORT but same DOMAIN.

### `script-2.py`
Advanced single-node experiment script for QuinkGL.

Includes advanced features:
- AffinityTopology — like-attracts-like peer selection
- FingerprintComputer — privacy-preserving data fingerprinting
- DataPolicy — manifest-driven collaboration/personalization policy
- PyTorchPersonalizedModel — FedRep/FedBN model split
- APFL adaptive mixing — personalized local/global weight blending
- PrototypeStore / FedPACCollaborator — optional prototype alignment

**How to use:**
1. Edit the CONFIG section in the script.
2. Run in one terminal: `python script-2.py`
3. For multi-node gossip, run in multiple terminals with different NODE_ID and PORT but same DOMAIN.

## Running Simulations

Install requirements from parent directory:
```bash
cd ..
pip install -e ".[dev]"
cd simulation-tests
```

Run individual scripts:
```bash
python script.py
# or
python script-1.py
# or
python script-2.py
```

## Important Notes

- With only one running node, you can verify node startup, local training, and topology selection, but aggregation differences are limited.
- To really compare aggregation behavior, run two or more nodes with the same DOMAIN and different NODE_ID / PORT values.
- The scripts use synthetic data and are designed for experimentation and testing, not production use.
