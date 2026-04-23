# TOFU Cache

The **Trust On First Use (TOFU) cache** stores the first-seen creator pubkey for each swarm ID.

## File Location

```
$XDG_STATE_HOME/quinkgl/tofu_creators.json
```

Default on Linux/macOS:
```
~/.local/state/quinkgl/tofu_creators.json
```

## Format

```json
{
  "<swarm_id_hex>": {
    "creator_pubkey": "ed25519:<64-hex>",
    "first_seen": "2026-04-23T12:00:00Z"
  }
}
```

## Atomic Writes

The cache is written atomically using **tmp + fsync + rename**:

1. Write to `tofu_creators.json.tmp`
2. `fsync` the file
3. `rename` over `tofu_creators.json`

This ensures the cache is never in a partially-written state, even if the process crashes.

## Tampering Risk

An attacker with filesystem access can modify the TOFU cache to accept a malicious creator.

**Mitigations:**
- Run QuinkGL with minimal filesystem privileges
- Set restrictive permissions on `$work_dir`
- Use `pinned` policy instead of `tofu` in high-security environments

## Manual Inspection

```bash
# View the cache
cat ~/.local/state/quinkgl/tofu_creators.json | python -m json.tool

# Remove a specific entry (forces re-TOFU on next join)
jq 'del(".acb115c412fa9187...")' tofu_creators.json > tofu_creators.json.new
mv tofu_creators.json.new tofu_creators.json
```

## Clearing the Cache

```bash
# Remove all TOFU entries
rm ~/.local/state/quinkgl/tofu_creators.json
```

**Warning:** Clearing the cache means the next manifest seen for any swarm will be trusted blindly.

## Migration Between Policies

| From | To | Action |
|---|---|---|
| `open` | `tofu` | No action; TOFU cache populates on first join |
| `tofu` | `pinned` | Extract pubkeys from cache, add to `--trusted-pubkey` |
| `pinned` | `tofu` | Not recommended; reduces security |
