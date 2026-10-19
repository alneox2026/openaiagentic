"""Reconciliation for reservations whose completed-turn event never arrived."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from common.billing import customer_wallet_document_id, nonnegative_int
from common.firestore import get_transaction_document_snapshot
from common.openai_usage_metadata import (
    aggregate_turn_usage,
    has_complete_turn_token_counts,
)
from common.schemas import TurnCompletedEvent
from services.agent_persistence_worker_v3.app.core.config import get_settings
from services.agent_persistence_worker_v3.app.core.errors import RetryableWorkerError
from services.agent_persistence_worker_v3.app.services.billing_settlement import (
    BillingSettlementService,
)
from services.agent_persistence_worker_v3.app.services.billing_ledger import (
    BillingLedgerRepository,
)
from services.agent_persistence_worker_v3.app.services.firestore_client import (
    get_firestore_client,
)
from services.agent_persistence_worker_v3.app.services.firestore_messages import (
    FirestoreMessagesRepository,
)
from services.agent_persistence_worker_v3.app.services.openai_turn_usage_hydrator import (
    OpenAITurnUsageHydrator,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BillingReconciliationResult:
    scanned_reservations: int
    settled_reservations: int
    released_reservations: int
    skipped_reservations: int


class BillingReconciliationService:
    """Settle finalized ledgers, or release genuinely orphaned expired holds."""

    def __init__(
        self,
        *,
        firestore_client_factory: Callable[[], Any] | None = None,
        settlement_service: BillingSettlementService | None = None,
        usage_hydrator: OpenAITurnUsageHydrator | None = None,
        ledger_repository: BillingLedgerRepository | None = None,
        messages_repository: FirestoreMessagesRepository | None = None,
    ) -> None:
        self._settings = get_settings()
        self._firestore_client_factory = firestore_client_factory or get_firestore_client
        self._settlement_service = settlement_service or BillingSettlementService(
            firestore_client_factory=self._firestore_client_factory,
        )
        # One non-blocking provider lookup per scheduled pass; the original
        # Pub/Sub worker already makes its short post-response retry attempts.
        self._usage_hydrator = usage_hydrator or OpenAITurnUsageHydrator(attempts=1)
        self._ledger_repository = ledger_repository or BillingLedgerRepository()
        self._messages_repository = messages_repository or FirestoreMessagesRepository()

    async def reconcile_expired(self) -> BillingReconciliationResult:
        return await asyncio.to_thread(self.reconcile_expired_sync)

    def reconcile_expired_sync(self) -> BillingReconciliationResult:
        client = self._firestore_client_factory()
        now = datetime.now(timezone.utc)
        reservations = self._reserved_reservations(client)
        scanned = settled = released = skipped = 0
        for reservation_snapshot in reservations:
            scanned += 1
            reservation = reservation_snapshot.to_dict() or {}
            if reservation.get("status") != "reserved":
                skipped += 1
                continue
            expired = _reservation_is_expired(reservation, now)
            turn_id = _required_string(reservation.get("turn_id"), "turn_id")
            ledger_snapshot = client.collection(
                self._settings.billing_ledger_collection
            ).document(turn_id).get()
            if ledger_snapshot.exists:
                ledger = ledger_snapshot.to_dict() or {}
                ledger = self._recover_openai_usage(client, ledger)
                if self._has_pending_openai_usage(ledger) and not expired:
                    skipped += 1
                    continue
                event = _event_from_ledger_and_reservation(
                    ledger,
                    reservation,
                )
                settlement = self._settlement_service.settle_if_required_sync(event)
                if settlement is None:
                    skipped += 1
                else:
                    settled += 1
                continue
            if expired and self._release_expired_reservation_sync(client, reservation, now):
                released += 1
            else:
                skipped += 1
        return BillingReconciliationResult(
            scanned_reservations=scanned,
            settled_reservations=settled,
            released_reservations=released,
            skipped_reservations=skipped,
        )

    def _reserved_reservations(self, client: Any):
        collection = client.collection(self._settings.billing_reservations_collection)
        batch_size = self._settings.billing_reconciliation_batch_size
        try:
            from google.cloud.firestore_v1.base_query import FieldFilter

            return (
                collection.where(filter=FieldFilter("status", "==", "reserved"))
                .order_by("expires_at")
                .limit(batch_size)
                .stream()
            )
        except (AttributeError, ModuleNotFoundError, ImportError):
            # Test doubles and older fakes may not implement Firestore queries.
            # Keep the same status filter and expiry ordering before limiting.
            candidates = []
            for snapshot in collection.stream():
                reservation = snapshot.to_dict() or {}
                expires_at = reservation.get("expires_at")
                if reservation.get("status") == "reserved":
                    if isinstance(expires_at, datetime):
                        candidates.append((_as_utc(expires_at), snapshot))
                    else:
                        candidates.append((datetime.max.replace(tzinfo=timezone.utc), snapshot))
            candidates.sort(key=lambda item: item[0])
            return [snapshot for _, snapshot in candidates[:batch_size]]

    def _recover_openai_usage(
        self,
        client: Any,
        ledger: dict[str, Any],
    ) -> dict[str, Any]:
        usage = ledger.get("usage")
        if not isinstance(usage, dict) or not self._has_pending_openai_usage(ledger):
            return ledger
        reports = usage.get("raw_turn_usage_reports")
        if not isinstance(reports, list):
            return self._mark_openai_usage_unpriced_pending(client, ledger, usage)

        session_id = str(ledger.get("session_id") or "").strip()
        if not session_id:
            return self._mark_openai_usage_unpriced_pending(client, ledger, usage)
        normalized_reports = [dict(report) for report in reports if isinstance(report, dict)]
        recovered_reports = self._usage_hydrator.hydrate(
            session_id=session_id,
            reports=normalized_reports,
        )
        token_reports = [
            report["usage"]
            for report in recovered_reports
            if has_complete_turn_token_counts(report.get("usage"))
        ]
        completed_count = _nonnegative_count(
            usage.get("completed_agent_turn_count"),
            default=len(recovered_reports),
        )
        missing_count = sum(
            not has_complete_turn_token_counts(report.get("usage"))
            for report in recovered_reports
        )
        if not recovered_reports and completed_count:
            missing_count = completed_count
        model_name = str(usage.get("pricing_model") or ledger.get("pricing_model") or "").strip()
        normalized = aggregate_turn_usage(
            token_reports,
            model_name=model_name,
            completed_turn_count=completed_count,
            missing_usage_turn_count=missing_count,
        )
        normalized.update(
            {
                "provider": "openai_agents_api",
                "billing_status": (
                    "pending_provider_usage"
                    if missing_count
                    else "processed_by_persistence_worker"
                ),
                "raw_turn_usage_reports": recovered_reports,
            }
        )
        if missing_count:
            normalized["known_cost_lower_bound_usd"] = normalized.get(
                "estimated_cost_usd"
            )
            normalized["estimated_cost_usd"] = None
            normalized["estimated_cost_complete"] = False
            normalized["cost_status"] = "unavailable"
        else:
            normalized["cost_status"] = (
                "estimated"
                if normalized.get("estimated_cost_usd") is not None
                else "unavailable"
            )

        should_persist = (
            recovered_reports != reports
            or usage.get("billing_status") != normalized.get("billing_status")
            or usage.get("usage_reporting_complete")
            != normalized.get("usage_reporting_complete")
            or usage.get("missing_usage_turn_count")
            != normalized.get("missing_usage_turn_count")
            or ledger.get("estimated_cost_usd") is not None
            or usage.get("estimated_cost_usd") != normalized.get("estimated_cost_usd")
        )
        if not should_persist:
            return ledger

        turn_id = _required_string(ledger.get("turn_id"), "turn_id")
        thread_id = str(ledger.get("thread_id") or "").strip()
        if thread_id:
            self._messages_repository.update_assistant_usage(
                client,
                thread_id=thread_id,
                turn_id=turn_id,
                usage=normalized,
            )
        self._ledger_repository.update_recovered_usage(client, turn_id, normalized)
        updated = dict(ledger)
        updated["usage"] = normalized
        updated["estimated_cost_usd"] = normalized.get("estimated_cost_usd")
        updated["cost_status"] = normalized.get("cost_status", "unavailable")
        return updated

    def _mark_openai_usage_unpriced_pending(
        self,
        client: Any,
        ledger: dict[str, Any],
        usage: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(usage)
        normalized["known_cost_lower_bound_usd"] = normalized.get(
            "estimated_cost_usd"
        )
        normalized["estimated_cost_usd"] = None
        normalized["estimated_cost_complete"] = False
        normalized["usage_reporting_complete"] = False
        normalized["billing_status"] = "pending_provider_usage"
        normalized["cost_status"] = "unavailable"
        if (
            ledger.get("estimated_cost_usd") is None
            and usage.get("estimated_cost_usd") is None
            and usage.get("billing_status") == "pending_provider_usage"
        ):
            return ledger
        turn_id = _required_string(ledger.get("turn_id"), "turn_id")
        thread_id = str(ledger.get("thread_id") or "").strip()
        if thread_id:
            self._messages_repository.update_assistant_usage(
                client,
                thread_id=thread_id,
                turn_id=turn_id,
                usage=normalized,
            )
        self._ledger_repository.update_recovered_usage(client, turn_id, normalized)
        updated = dict(ledger)
        updated["usage"] = normalized
        updated["estimated_cost_usd"] = None
        updated["cost_status"] = "unavailable"
        return updated

    @staticmethod
    def _has_pending_openai_usage(ledger: dict[str, Any]) -> bool:
        usage = ledger.get("usage")
        if not isinstance(usage, dict):
            return False
        provider = usage.get("provider") or ledger.get("provider")
        if provider != "openai_agents_api":
            return False
        missing_count = usage.get("missing_usage_turn_count", 0)
        if isinstance(missing_count, int) and not isinstance(missing_count, bool):
            if missing_count > 0:
                return True
        reports = usage.get("raw_turn_usage_reports")
        if usage.get("usage_reporting_complete") is False:
            return True
        return isinstance(reports, list) and any(
            isinstance(report, dict)
            and not has_complete_turn_token_counts(report.get("usage"))
            for report in reports
        )

    def _release_expired_reservation_sync(
        self,
        client: Any,
        reservation: dict[str, Any],
        released_at: datetime,
    ) -> bool:
        turn_id = _required_string(reservation.get("turn_id"), "turn_id")
        billing_subject_id = _required_string(
            reservation.get("billing_subject_id"),
            "billing_subject_id",
        )
        owner_uid = _required_string(reservation.get("owner_uid"), "owner_uid")
        reservation_ref = client.collection(
            self._settings.billing_reservations_collection
        ).document(turn_id)
        wallet_ref = client.collection(self._settings.wallets_collection).document(
            customer_wallet_document_id(billing_subject_id)
        )
        transaction_ref = client.collection(
            self._settings.wallet_transactions_collection
        ).document(f"reservation_expired_{turn_id}")

        def operation(transaction: Any) -> bool:
            # Keep all reads before writes for Firestore transaction validity.
            current_reservation_snapshot = get_transaction_document_snapshot(transaction, reservation_ref)
            wallet_snapshot = get_transaction_document_snapshot(transaction, wallet_ref)
            transaction_snapshot = get_transaction_document_snapshot(transaction, transaction_ref)
            if transaction_snapshot.exists or not current_reservation_snapshot.exists:
                return False
            current_reservation = current_reservation_snapshot.to_dict() or {}
            if current_reservation.get("status") != "reserved":
                return False
            if not wallet_snapshot.exists:
                raise RetryableWorkerError(
                    f"Wallet is missing for expired reservation {turn_id}."
                )
            wallet = wallet_snapshot.to_dict() or {}
            if (
                wallet.get("status") != "active"
                or wallet.get("billing_subject_id") != billing_subject_id
                or wallet.get("owner_uid") != owner_uid
            ):
                raise RetryableWorkerError(
                    f"Wallet does not match expired reservation {turn_id}."
                )
            try:
                reserved_amount_nanos = nonnegative_int(
                    current_reservation.get("reserved_amount_nanos"),
                    field_name="reserved_amount_nanos",
                )
                available_credit_nanos = int(wallet.get("available_credit_nanos", 0))
                reserved_credit_nanos = nonnegative_int(
                    wallet.get("reserved_credit_nanos"),
                    field_name="reserved_credit_nanos",
                )
            except ValueError as exc:
                raise RetryableWorkerError(
                    f"Wallet data is invalid for expired reservation {turn_id}."
                ) from exc
            if reserved_credit_nanos < reserved_amount_nanos:
                raise RetryableWorkerError(
                    f"Wallet reserved credit is invalid for expired reservation {turn_id}."
                )
            transaction.create(
                transaction_ref,
                {
                    "schema_version": 1,
                    "transaction_id": f"reservation_expired_{turn_id}",
                    "transaction_type": "reservation_expiry_release",
                    "status": "released",
                    "billing_subject_id": billing_subject_id,
                    "owner_uid": owner_uid,
                    "wallet_document_id": customer_wallet_document_id(billing_subject_id),
                    "currency": current_reservation.get("currency", "USD"),
                    "turn_id": turn_id,
                    "request_id": current_reservation.get("request_id"),
                    "agent_id": current_reservation.get("agent_id"),
                    "reservation_id": turn_id,
                    "reserved_amount_nanos": reserved_amount_nanos,
                    "amount_nanos": 0,
                    "released_amount_nanos": reserved_amount_nanos,
                    "created_at": released_at,
                    "reason": "completed_turn_not_received_before_reservation_expiry",
                },
            )
            transaction.update(
                wallet_ref,
                {
                    "available_credit_nanos": available_credit_nanos + reserved_amount_nanos,
                    "reserved_credit_nanos": reserved_credit_nanos - reserved_amount_nanos,
                    "updated_at": released_at,
                    "last_reservation_release_at": released_at,
                },
            )
            transaction.update(
                reservation_ref,
                {
                    "status": "expired_released",
                    "released_at": released_at,
                    "released_amount_nanos": reserved_amount_nanos,
                    "release_reason": "completed_turn_not_received_before_reservation_expiry",
                },
            )
            return True

        return _run_firestore_transaction(client, operation)


def _event_from_ledger_and_reservation(
    ledger: dict[str, Any],
    reservation: dict[str, Any],
) -> TurnCompletedEvent:
    return TurnCompletedEvent(
        event_id=_required_string(ledger.get("event_id"), "event_id"),
        turn_id=_required_string(ledger.get("turn_id"), "turn_id"),
        agent_id=_required_string(ledger.get("agent_id"), "agent_id"),
        user_id=_required_string(ledger.get("uid"), "uid"),
        thread_id=_required_string(ledger.get("thread_id"), "thread_id"),
        session_id=_required_string(ledger.get("session_id"), "session_id"),
        user_message="",
        assistant_message="",
        created_at=_required_datetime(ledger.get("created_at"), "created_at"),
        metadata={
            "billing": {
                "reservation_id": reservation.get("reservation_id"),
                "billing_subject_id": reservation.get("billing_subject_id"),
            }
        },
    )


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetryableWorkerError(f"Billing reconciliation field {field_name} is required.")
    return value.strip()


def _reservation_is_expired(reservation: dict[str, Any], now: datetime) -> bool:
    expires_at = reservation.get("expires_at")
    return isinstance(expires_at, datetime) and _as_utc(expires_at) <= now


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _nonnegative_count(value: Any, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return max(0, default)
    return value


def _required_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise RetryableWorkerError(f"Billing reconciliation field {field_name} is required.")
    return value


def _run_firestore_transaction(client: Any, operation: Callable[[Any], bool]) -> bool:
    from google.cloud import firestore

    transaction = client.transaction()

    @firestore.transactional
    def run(transaction: Any) -> bool:
        return operation(transaction)

    return run(transaction)
