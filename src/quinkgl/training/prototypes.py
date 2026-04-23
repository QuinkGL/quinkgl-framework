"""
FedProto / FedPAC — Prototype-based alignment for domain-aware collaboration.

This layer is OPTIONAL and disabled by default.
Phase 6e is experimental; it must not be activated until 6a–6d are complete.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


@dataclass
class ClassPrototype:
    label: str
    embedding: np.ndarray
    sample_count: int
    variance: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "label": self.label,
            "embedding": self.embedding.tolist(),
            "sample_count": self.sample_count,
        }
        if self.variance is not None:
            d["variance"] = self.variance.tolist()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassPrototype":
        return cls(
            label=data["label"],
            embedding=np.array(data["embedding"]),
            sample_count=data["sample_count"],
            variance=np.array(data["variance"]) if "variance" in data else None,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "ClassPrototype":
        return cls.from_dict(json.loads(json_str))


class PrototypeStore:
    """Manages local and global prototypes for FedProto alignment."""

    def __init__(self):
        self.local_prototypes: Dict[str, ClassPrototype] = {}
        self.global_prototypes: Dict[str, ClassPrototype] = {}

    def compute_local_prototypes(
        self,
        features_by_label: Dict[str, List[np.ndarray]],
    ) -> Dict[str, ClassPrototype]:
        self.local_prototypes = {}
        for label, features_list in features_by_label.items():
            if not features_list:
                continue
            arr = np.stack(features_list)
            centroid = np.mean(arr, axis=0)
            variance = np.var(arr, axis=0)
            self.local_prototypes[label] = ClassPrototype(
                label=label,
                embedding=centroid,
                sample_count=len(features_list),
                variance=variance,
            )
        return self.local_prototypes

    def merge_global_prototypes(
        self,
        peer_prototypes: Dict[str, List[ClassPrototype]],
    ) -> Dict[str, ClassPrototype]:
        label_accum: Dict[str, List[Tuple[np.ndarray, int]]] = {}
        for peer_id, prototypes in peer_prototypes.items():
            for proto in prototypes:
                label_accum.setdefault(proto.label, []).append(
                    (proto.embedding, proto.sample_count)
                )
        # Don't reset global_prototypes - accumulate incrementally
        for label, entries in label_accum.items():
            total_samples = sum(n for _, n in entries)
            weighted_sum = sum(emb * n for emb, n in entries)
            if label in self.global_prototypes:
                # Merge with existing global prototype using sample-weighted average
                existing = self.global_prototypes[label]
                combined_samples = existing.sample_count + total_samples
                combined_embedding = (
                    existing.embedding * existing.sample_count + weighted_sum
                ) / combined_samples
                self.global_prototypes[label] = ClassPrototype(
                    label=label,
                    embedding=combined_embedding,
                    sample_count=combined_samples,
                )
            else:
                self.global_prototypes[label] = ClassPrototype(
                    label=label,
                    embedding=weighted_sum / total_samples,
                    sample_count=total_samples,
                )
        return self.global_prototypes

    def prototype_alignment_loss(self) -> float:
        loss = 0.0
        count = 0
        for label in self.local_prototypes:
            if label in self.global_prototypes:
                local_emb = self.local_prototypes[label].embedding
                global_emb = self.global_prototypes[label].embedding
                loss += float(np.mean((local_emb - global_emb) ** 2))
                count += 1
        return loss / max(count, 1)

    def local_prototypes_to_json(self) -> str:
        data = [p.to_dict() for p in self.local_prototypes.values()]
        return json.dumps(data)

    @staticmethod
    def parse_peer_prototypes(json_str: str) -> List[ClassPrototype]:
        """TASK-043: Removed unused peer_id parameter."""
        data = json.loads(json_str)
        return [ClassPrototype.from_dict(d) for d in data]


class FedPACCollaborator:
    """Optimal classifier combination from similar peers (FedPAC).

    Computes discrepancy between peers based on their fingerprints
    and combines classifier heads with optimal weights.
    """

    def compute_discrepancy(
        self,
        my_prototypes: Dict[str, ClassPrototype],
        peer_prototypes: Dict[str, Dict[str, ClassPrototype]],
    ) -> Dict[str, float]:
        discrepancy: Dict[str, float] = {}
        for peer_id, prototypes in peer_prototypes.items():
            total_diff = 0.0
            shared = 0
            for label in my_prototypes:
                if label in prototypes:
                    diff = np.linalg.norm(
                        my_prototypes[label].embedding - prototypes[label].embedding
                    )
                    total_diff += diff
                    shared += 1
            # Return infinity for zero-overlap peers (they should be filtered out)
            discrepancy[peer_id] = total_diff / max(shared, 1) if shared > 0 else float('inf')
        return discrepancy

    def compute_combination_weights(
        self,
        discrepancy: Dict[str, float],
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """TASK-027: Added parameter validation."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not discrepancy:
            return {}
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        # Filter out zero-overlap peers (discrepancy = inf)
        valid_peers = {pid: d for pid, d in discrepancy.items() if d != float('inf')}
        if not valid_peers:
            # All peers have zero overlap, return uniform weights
            n = len(discrepancy)
            return {pid: 1.0 / n for pid in discrepancy}
        exp_scores: Dict[str, float] = {}
        for pid, d in valid_peers.items():
            exp_scores[pid] = np.exp(-d / temperature)
        total = sum(exp_scores.values())
        if total == 0:
            n = len(valid_peers)
            return {pid: 1.0 / n for pid in valid_peers}
        return {pid: s / total for pid, s in exp_scores.items()}
