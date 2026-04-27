from __future__ import annotations

import hashlib
import secrets
import string
from dataclasses import dataclass
from datetime import datetime, timedelta


DEFAULT_DASHBOARD_CODE_TTL_SECONDS = 600
DEFAULT_VIEWER_TOKEN_TTL_SECONDS = 12 * 60 * 60


@dataclass(frozen=True)
class DashboardViewerScope:
    swarm_id: str
    issued_from_node_id: str | None
    expires_at: datetime

    def to_dict(self) -> dict[str, str | None]:
        return {
            "swarm_id": self.swarm_id,
            "issued_from_node_id": self.issued_from_node_id,
            "expires_at": self.expires_at.isoformat(),
        }


@dataclass
class _DashboardCodeRecord:
    code_hash: str
    swarm_id: str
    issued_from_node_id: str | None
    expires_at: datetime
    redeemed: bool = False


class DashboardAccessRegistry:
    """In-memory dashboard code and viewer-token registry."""

    def __init__(
        self,
        *,
        code_ttl_seconds: int = DEFAULT_DASHBOARD_CODE_TTL_SECONDS,
        viewer_token_ttl_seconds: int = DEFAULT_VIEWER_TOKEN_TTL_SECONDS,
    ):
        self.code_ttl_seconds = code_ttl_seconds
        self.viewer_token_ttl_seconds = viewer_token_ttl_seconds
        self._codes: dict[str, _DashboardCodeRecord] = {}
        self._viewer_tokens: dict[str, DashboardViewerScope] = {}

    @staticmethod
    def _hash_secret(value: str) -> str:
        return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _new_dashboard_code() -> str:
        alphabet = string.ascii_uppercase + string.digits
        first = "".join(secrets.choice(alphabet) for _ in range(4))
        second = "".join(secrets.choice(alphabet) for _ in range(4))
        return f"QGL-{first}-{second}"

    def create_code(
        self,
        *,
        swarm_id: str,
        issued_from_node_id: str | None = None,
        now: datetime | None = None,
    ) -> tuple[str, DashboardViewerScope]:
        if not isinstance(swarm_id, str) or not swarm_id.strip():
            raise ValueError("swarm_id is required")
        now = now or datetime.now()
        code = self._new_dashboard_code()
        expires_at = now + timedelta(seconds=self.code_ttl_seconds)
        record = _DashboardCodeRecord(
            code_hash=self._hash_secret(code),
            swarm_id=swarm_id,
            issued_from_node_id=issued_from_node_id,
            expires_at=expires_at,
        )
        self._codes[record.code_hash] = record
        return code, DashboardViewerScope(
            swarm_id=swarm_id,
            issued_from_node_id=issued_from_node_id,
            expires_at=expires_at,
        )

    def redeem_code(
        self,
        code: str,
        *,
        now: datetime | None = None,
    ) -> tuple[str, DashboardViewerScope] | None:
        if not isinstance(code, str) or not code.strip():
            return None
        now = now or datetime.now()
        record = self._codes.get(self._hash_secret(code.strip().upper()))
        if record is None or record.redeemed or record.expires_at <= now:
            return None
        record.redeemed = True
        token = "qgl_view_" + secrets.token_urlsafe(32)
        scope = DashboardViewerScope(
            swarm_id=record.swarm_id,
            issued_from_node_id=record.issued_from_node_id,
            expires_at=now + timedelta(seconds=self.viewer_token_ttl_seconds),
        )
        self._viewer_tokens[self._hash_secret(token)] = scope
        return token, scope

    def resolve_viewer_token(
        self,
        token: str | None,
        *,
        now: datetime | None = None,
    ) -> DashboardViewerScope | None:
        if not token:
            return None
        now = now or datetime.now()
        scope = self._viewer_tokens.get(self._hash_secret(token))
        if scope is None or scope.expires_at <= now:
            return None
        return scope
