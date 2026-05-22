from __future__ import annotations

import threading
from datetime import datetime, timezone

from pymongo import MongoClient

import certifi


class DailyRetrainScheduler:
    """Runs a background thread that fires once per day at midnight UTC."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="foodintel-daily-scheduler", daemon=True
        )
        self._thread.start()
        print("[Scheduler] Daily retraining scheduler started.")

    def stop(self) -> None:
        self._stop_event.set()

    def _seconds_until_midnight_utc(self) -> float:
        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if now >= midnight:
            from datetime import timedelta
            midnight = midnight + timedelta(days=1)
        return (midnight - now).total_seconds()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            wait_seconds = self._seconds_until_midnight_utc()
            print(f"[Scheduler] Next daily retrain check in {wait_seconds/3600:.1f} hours.")
            triggered = self._stop_event.wait(timeout=wait_seconds)
            if triggered:
                break
            self._run_daily_check()

    def _run_daily_check(self) -> None:
        from app.config.settings import get_settings
        from app.services.retraining_service import retraining_service

        print("[Scheduler] Running daily approved-feedback retraining check.")
        settings = get_settings()
        try:
            client_kwargs = {}
            if "mongodb.net" in settings.mongodb_uri.lower():
                client_kwargs["tlsCAFile"] = certifi.where()
            client = MongoClient(settings.mongodb_uri, **client_kwargs)
            try:
                collection = client[settings.mongodb_db_name]["prediction_feedback"]
                approved_count = collection.count_documents(
                    {"status": {"$in": ["approved"]}}
                )
            finally:
                client.close()
        except Exception as exc:
            print(f"[Scheduler] Could not count approved feedback: {exc}")
            return

        print(f"[Scheduler] Found {approved_count} approved feedback samples.")
        decision = retraining_service.request_retraining(approved_count)
        print(f"[Scheduler] Retraining decision: {decision.status} — {decision.message}")


daily_scheduler = DailyRetrainScheduler()
