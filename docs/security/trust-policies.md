# Trust Policies

QuinkGL provides three trust policies that govern how peers evaluate manifest creators.

## Policy Comparison

| Policy | Signed Required | Trust Basis | Use Case |
|---|---|---|---|
| `open` | No | None | Rapid prototyping, private networks |
| `tofu` | Optional | First-seen creator cached | Small consortiums with churn |
| `pinned` | Yes | Pre-configured pubkey whitelist | Production deployments |

## open

Accept any manifest regardless of signature.

```bash
quinkgl run --manifest swarm.qgl --data ./data --trust-policy open
```

**Behavior:**
- Signed manifests: signature verified but not required
- Unsigned manifests: accepted silently
- Creator changes: ignored

**When to use:**
- Internal test networks
- Rapid prototyping
- Workshops and demos

**Risk:** Any peer can publish a malicious manifest and join the swarm.

## tofu (Trust On First Use)

Cache the first creator pubkey seen for each swarm ID. Reject manifests from different creators.

```bash
quinkgl run --manifest swarm.qgl --data ./data --trust-policy tofu
```

**Behavior:**
- First encounter: cache `creator_pubkey` for this `swarm_id`
- Re-encounter with same creator: accepted
- Re-encounter with different creator: `ERR_TRUST_TOFU_CONFLICT`

**Cache location:**
```
$work_dir/tofu_creators.json
```

Example cache:
```json
{
  "acb115c412fa9187...": {
    "creator_pubkey": "ed25519:5a5e0985...",
    "first_seen": "2026-04-23T12:00:00Z"
  }
}
```

**When to use:**
- Small consortiums where membership changes rarely
- Situations where pre-sharing pubkeys is inconvenient

**Risk:** First-seen manifest might be from an attacker if the network is compromised during initial deployment.

## pinned

Only accept manifests from explicitly trusted creators.

```bash
quinkgl run \
  --manifest swarm.qgl \
  --data ./data \
  --trust-policy pinned \
  --trusted-pubkey ed25519:5a5e0985...
```

**Behavior:**
- Unsigned manifests: rejected (`ERR_NODE_UNSIGNED_MANIFEST_REJECTED`)
- Signed by trusted creator: accepted
- Signed by unknown creator: rejected (`ERR_CREATOR_NOT_TRUSTED`)

**When to use:**
- Production deployments
- Regulated environments (healthcare, finance)
- Any scenario where creator identity matters

**Configuration:**
```python
from quinkgl import GossipNode

node = GossipNode(
    node_id="alice",
    manifest=manifest,
    model=model,
    trust_policy="pinned",
    trusted_creator_pubkeys={
        bytes.fromhex("5a5e0985..."),
    },
)
```

## Choosing a Policy

```
Are you in production?          → pinned
Are you in a trusted network?   → tofu
Are you just testing?           → open
```

## Error Codes

| Code | Policy | Meaning |
|---|---|---|
| `ERR_TRUST_POLICY_VIOLATION` | Any | General trust policy violation |
| `ERR_TRUST_TOFU_CONFLICT` | tofu | Cached creator differs from manifest |
| `ERR_NODE_UNSIGNED_MANIFEST_REJECTED` | pinned | Manifest lacks signature |
| `ERR_CREATOR_NOT_TRUSTED` | pinned | Creator not in trusted set |
