# `quinkgl init`

Scaffold a user peer-script project from a template.

## Synopsis

```bash
quinkgl init --output-dir <path> [--manifest <qgl>] [--template <name>] [--framework <name>]
```

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | **required** | Destination directory (must not exist) |
| `--manifest` | — | `.qgl` file to embed in the scaffold |
| `--template` | `minimal` | `minimal`, `pytorch-vision`, `pytorch-tabular`, or `custom` |
| `--framework` | manifest's framework or `pytorch` | Override model framework hint |

## Templates

| Template | Contents |
|----------|----------|
| `minimal` | Stub `build_model` and `build_loaders`, `pyproject.toml`, tests |
| `pytorch-vision` | CNN skeleton for image classification |
| `pytorch-tabular` | MLP skeleton for tabular data |
| `custom` | Bare boilerplate with no framework assumptions |

## Examples

```bash
quinkgl init --output-dir my-peer --manifest demo.qgl --template pytorch-vision
```

## Generated Files

```
my-peer/
  peer_script.py       # build_model, build_loaders stubs
  tests/
    test_peer.py       # sanity checks
  pyproject.toml       # dependencies
  <manifest>.qgl       # copied if --manifest provided
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Validation error |
| `2` | I/O error (directory already exists) |

## See Also

- [Peer Scripts](../user-guide/peer-script.md)
- [Tutorial T4](../tutorials/T4/index.md)
