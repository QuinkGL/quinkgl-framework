from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class TelemetryTokenRecord:
    swarm_id: str
    token_hash: str
    name: str = ""
    revoked: bool = False


class TelemetryTokenRegistry:
    def __init__(
        self,
        records: Iterable[TelemetryTokenRecord] = (),
        *,
        path: str | Path | None = None,
    ):
        self.path = Path(path) if path is not None else None
        self._records = {
            record.token_hash: record
            for record in records
            if not record.revoked
        }
        self._all_records = list(records)

    @staticmethod
    def hash_token(token: str) -> str:
        return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()

    @classmethod
    def from_plain_tokens(cls, entries: Iterable[dict[str, Any]]) -> "TelemetryTokenRegistry":
        records = []
        for entry in entries:
            token = entry.get("token")
            token_hash = entry.get("token_hash")
            if token:
                token_hash = cls.hash_token(str(token))
            if not token_hash:
                raise ValueError("telemetry token entry requires token or token_hash")
            swarm_id = entry.get("swarm_id")
            if not isinstance(swarm_id, str) or not swarm_id:
                raise ValueError("telemetry token entry requires swarm_id")
            records.append(
                TelemetryTokenRecord(
                    swarm_id=swarm_id,
                    token_hash=str(token_hash),
                    name=str(entry.get("name") or ""),
                    revoked=bool(entry.get("revoked", False)),
                )
            )
        return cls(records)

    @classmethod
    def from_file(cls, path: str | Path, *, missing_ok: bool = False) -> "TelemetryTokenRegistry":
        p = Path(path)
        if missing_ok and not p.exists():
            return cls(path=p)
        raw = p.read_text(encoding="utf-8")
        if not raw.strip():
            return cls(path=p)
        data = json.loads(raw)
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = data.get("tokens", [])
        else:
            raise ValueError("telemetry token file must contain an object or list")
        if not isinstance(entries, list):
            raise ValueError("telemetry token file tokens must be a list")
        registry = cls.from_plain_tokens(entries)
        registry.path = p
        return registry

    def resolve(self, token: str | None) -> TelemetryTokenRecord | None:
        if not token:
            return None
        return self._records.get(self.hash_token(token))

    def create_token(self, *, swarm_id: str, name: str = "") -> str:
        if not isinstance(swarm_id, str) or not swarm_id:
            raise ValueError("swarm_id is required")
        token = "qgl_live_" + secrets.token_urlsafe(32)
        record = TelemetryTokenRecord(
            swarm_id=swarm_id,
            token_hash=self.hash_token(token),
            name=name,
        )
        self._records[record.token_hash] = record
        self._all_records.append(record)
        self.persist()
        return token

    def persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tokens": [
                {
                    "swarm_id": record.swarm_id,
                    "token_hash": record.token_hash,
                    "name": record.name,
                    "revoked": record.revoked,
                }
                for record in self._all_records
            ]
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
