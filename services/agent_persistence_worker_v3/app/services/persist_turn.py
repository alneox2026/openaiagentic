"""Turn persistence orchestration for Pub/Sub-delivered events."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from common.openai_usage_metadata import aggregate_turn_usage
from common.schemas import TurnCompletedEvent
from services.agent_persistence_worker_v3.app.core.errors import RetryableWorkerError
from services.agent_persistence_worker_v3.app.services.firestore_client import (
    get_firestore_client,
)
from services.agent_persistence_worker_v3.app.services.billing_ledger import (
    BillingLedgerRepository,
)
from services.agent_persistence_worker_v3.app.services.billing_settlement import (
    BillingSettlementService,
)
from services.agent_persistence_worker_v3.app.services.firestore_messages import (
    FirestoreMessagesRepository,
)
from services.agent_persistence_worker_v3.app.services.firestore_threads import (
    FirestoreThreadsRepository,
)
from services.agent_persistence_worker_v3.app.services.idempotency import IdempotencyStore
from services.agent_persistence_worker_v3.app.services.openai_turn_usage_hydrator import (
    OpenAITurnUsageHydrator,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersistTurnResult:
    event_id: str
    thread_id: str
    persisted: bool
    ignored_reason: str | None = None
    billing_settlement_status: str | None = None


class PersistTurnService:
    def __init__(
        self,
        idempotency_store: IdempotencyStore | None = None,
        billing_ledger_repository: BillingLedgerRepository | None = None,
        billing_settlement_service: BillingSettlementService | None = None,
        threads_repository: FirestoreThreadsRepository | None = None,
        messages_repository: FirestoreMessagesRepository | None = None,
        firestore_client_factory: Callable[[], Any] | None = None,
        openai_turn_usage_hydrator: OpenAITurnUsageHydrator | None = None,
    ) -> None:
        self.idempotency_store = idempotency_store or IdempotencyStore()
        self.billing_ledger_repository = (
            billing_ledger_repository or BillingLedgerRepository()
        )
        self.billing_settlement_service = (
            billing_settlement_service or BillingSettlementService()
        )
        self.threads_repository = (
            threads_repository or FirestoreThreadsRepository()
        )
        self.messages_repository = (
            messages_repository or FirestoreMessagesRepository()
        )
        self.firestore_client_factory = firestore_client_factory or get_firestore_client
        self.openai_turn_usage_hydrator = (
            openai_turn_usage_hydrator or OpenAITurnUsageHydrator()
        )

    async def persist(self, event: TurnCompletedEvent) -> PersistTurnResult:
        return await asyncio.to_thread(self._persist_sync, event)

    def _persist_sync(self, event: TurnCompletedEvent) -> PersistTurnResult:
        try:
            event = self._normalize_openai_usage_after_response(event)
        except Exception as exc:
            raise RetryableWorkerError(
                f"Failed to calculate OpenAI turn-token billing for {event.turn_id}: {exc}"
            ) from exc
        try:
            client = self.firestore_client_factory()
            existing_thread = self.threads_repository.load_existing(
                client,
                event.thread_id,
            )
            ignored_reason = self.threads_repository.validate_existing_for_turn(
                existing_thread,
                event,
            )
            if ignored_reason:
                return self._persist_idempotency_only(
                    client,
                    event,
                    ignored_reason=ignored_reason,
                )
            batch = client.batch()
            self.idempotency_store.add_create_to_batch(batch, client, event)
            self.billing_ledger_repository.add_create_to_batch(batch, client, event)
            self.threads_repository.add_upsert_to_batch(
                batch,
                client,
                event,
                existing_thread=existing_thread,
            )
            self.messages_repository.add_turn_messages_to_batch(
                batch,
                client,
                event,
            )
            batch.commit()
        except Exception as exc:
            if self._is_conflict_error(exc):
                persisted = False
            else:
                raise RetryableWorkerError(
                    f"Failed to persist turn event {event.event_id}: {exc}"
                ) from exc
        else:
            persisted = True

        try:
            settlement = self.billing_settlement_service.settle_if_required_sync(event)
        except Exception as exc:
            if isinstance(exc, RetryableWorkerError):
                raise
            raise RetryableWorkerError(
                f"Failed to settle billing for turn {event.turn_id}: {exc}"
            ) from exc

        return PersistTurnResult(
            event_id=event.event_id,
            thread_id=event.thread_id,
            persisted=persisted,
            billing_settlement_status=settlement.status if settlement else None,
        )

    def _normalize_openai_usage_after_response(
        self,
        event: TurnCompletedEvent,
    ) -> TurnCompletedEvent:
        metering = event.infrastructure_metering
        usage = event.usage if isinstance(event.usage, dict) else {}
        raw_reports = usage.get("raw_turn_usage_reports")
        if (
            metering is None
            or not isinstance(raw_reports, list)
        ):
            return event

        reports = self.openai_turn_usage_hydrator.hydrate(
            session_id=event.session_id,
            reports=[report for report in raw_reports if isinstance(report, dict)],
        )
        token_reports = [
            report["usage"]
            for report in reports
            if isinstance(report.get("usage"), dict)
        ]
        completed_count = self._nonnegative_count(
            usage.get("completed_agent_turn_count"),
            default=len(reports),
        )
        missing_count = sum(
            not isinstance(report.get("usage"), dict) for report in reports
        )
        model_name = str(usage.get("pricing_model") or "").strip()
        normalized = aggregate_turn_usage(
            token_reports,
            model_name=model_name,
            completed_turn_count=completed_count,
            missing_usage_turn_count=missing_count,
        )
        normalized.update(
            {
                "provider": "openai_agents_api",
                "billing_status": "processed_by_persistence_worker",
                "raw_turn_usage_reports": reports,
            }
        )
        if missing_count:
            LOGGER.warning(
                "OpenAI token-rate billing has usage missing for %s of %s agent turn(s).",
                missing_count,
                completed_count,
            )
        return event.model_copy(update={"usage": normalized})

    @staticmethod
    def _nonnegative_count(value: Any, *, default: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return max(0, default)
        return value

    def _persist_idempotency_only(
        self,
        client: Any,
        event: TurnCompletedEvent,
        *,
        ignored_reason: str,
    ) -> PersistTurnResult:
        batch = client.batch()
        self.idempotency_store.add_create_to_batch(batch, client, event)
        try:
            batch.commit()
        except Exception as exc:
            if self._is_conflict_error(exc):
                return PersistTurnResult(
                    event_id=event.event_id,
                    thread_id=event.thread_id,
                    persisted=False,
                    ignored_reason=ignored_reason,
                )
            raise RetryableWorkerError(
                f"Failed to record ignored turn event {event.event_id}: {exc}"
            ) from exc
        return PersistTurnResult(
            event_id=event.event_id,
            thread_id=event.thread_id,
            persisted=False,
            ignored_reason=ignored_reason,
        )

    def _is_conflict_error(self, exc: Exception) -> bool:
        if exc.__class__.__name__ == "Conflict":
            return True
        try:
            from google.cloud import exceptions as cloud_exceptions

            return isinstance(exc, cloud_exceptions.Conflict)
        except ModuleNotFoundError:
            return False
