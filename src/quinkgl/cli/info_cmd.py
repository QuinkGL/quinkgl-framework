# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""quinkgl info — framework introspection."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import _SubParsersAction

import quinkgl


def build_parser(sub: _SubParsersAction) -> None:
    sub.add_parser("info", help="Framework version + registered strategies")


def run(args: argparse.Namespace) -> int:
    data = {
        "version": quinkgl.__version__,
        "manifest_schema": quinkgl.MANIFEST_SCHEMA_VERSION,
        "python": platform.python_version(),
        "ipv8": _get_ipv8_version(),
        "cryptography": _get_crypto_version(),
        "registered_aggregations": _list_aggregations(),
        "registered_topologies": _list_topologies(),
        "model_frameworks": _list_frameworks(),
    }
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print(f"QuinkGL:            {data['version']}")
        print(f"Manifest schema:    v{data['manifest_schema']}")
        print(f"Python:             {data['python']}")
        print(f"IPv8:               {data['ipv8']}")
        print(f"cryptography:       {data['cryptography']}")
        print()
        print(f"Registered aggregations: {', '.join(data['registered_aggregations'])}")
        print(f"Registered topologies:   {', '.join(data['registered_topologies'])}")
        print(f"Model frameworks:        {', '.join(data['model_frameworks'])}")
    return 0


def _get_ipv8_version() -> str:
    try:
        from importlib.metadata import version
        return version("pyipv8")
    except Exception:
        try:
            from ipv8 import __version__ as v
            return v
        except Exception:
            return "unknown"


def _get_crypto_version() -> str:
    try:
        import cryptography
        return cryptography.__version__
    except Exception:
        return "not installed"


def _list_aggregations() -> list[str]:
    # TODO: discover from quinkgl.aggregation registry
    return ["FedAvg", "FedProx", "FedAvgM", "TrimmedMean", "Krum", "MultiKrum",
            "StalenessWeightedFedAvg", "EntropyWeightedAvg", "Scaffold"]


def _list_topologies() -> list[str]:
    return ["RandomTopology", "CyclonTopology", "AffinityTopology"]


def _list_frameworks() -> list[str]:
    frameworks = ["pytorch"]
    if getattr(quinkgl, "_tensorflow_available", False):
        frameworks.append("tensorflow")
    frameworks.append("custom")
    return frameworks
