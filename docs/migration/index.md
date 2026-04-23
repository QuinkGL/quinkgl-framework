# Migration Guide

Guides for upgrading across breaking changes.

## Current Versions

| From | To | Guide |
|---|---|---|
| v0.3.x | v0.4.0 | Manifest schema v2 → v3 |

## v0.3.x → v0.4.0

### What Changed

- `MANIFEST_SCHEMA_VERSION` bumped from 2 to 3
- New required fields: `name`, `task`, `model`
- New optional fields: `byzantine`, `round_limit`, `bootstrap_peers`, `tracker_urls`
- CLI `quinkgl` introduced

### Before/After

**Before (v0.3):**
```python
from quinkgl import GossipNode
node = GossipNode(
    node_id="alice",
    domain="health",
    model=model,
)
```

**After (v0.4):**
```python
from quinkgl import GossipNode
from quinkgl.manifest import SwarmManifest

manifest = SwarmManifest.from_file("swarm.qgl")
node = GossipNode(
    node_id="alice",
    manifest=manifest,
    model=model,
)
```

### Migration Checklist

- [ ] Create a v3 manifest using `quinkgl manifest create`
- [ ] Verify the manifest with `quinkgl manifest verify`
- [ ] Update peer startup scripts to use `manifest=` instead of `domain=`
- [ ] Pin `quinkgl>=0.4,<0.5` in `pyproject.toml`
- [ ] Run `pytest` against the updated scripts
