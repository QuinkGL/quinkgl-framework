# `quinkgl info`

Print framework version and registered strategies.

## Synopsis

```bash
quinkgl info
```

## Examples

```bash
$ quinkgl info
QuinkGL:            0.3.4
Manifest schema:    v4
Python:             3.12.2
IPv8:               2.14.0
cryptography:       42.0.5

Registered aggregations: FedAvg, FedProx, FedAvgM, TrimmedMean, Krum, MultiKrum, StalenessWeightedFedAvg, EntropyWeightedAvg, Scaffold
Registered topologies:   RandomTopology, CyclonTopology, AffinityTopology
Model frameworks:        pytorch, custom
```

JSON output:

```bash
quinkgl --json info
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |

## See Also

- [CLI Reference Overview](../reference/cli-reference.md)
