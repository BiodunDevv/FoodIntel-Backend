from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import certifi
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase

from app.config.settings import Settings


class MongoManager:
    def __init__(self) -> None:
        self.client: AsyncIOMotorClient | None = None
        self.database: AsyncIOMotorDatabase | None = None
        self.connected = False
        self.last_error: str | None = None

    async def connect(self, settings: Settings) -> None:
        try:
            client_kwargs: dict[str, Any] = {}
            uri_lower = settings.mongodb_uri.lower()
            if "mongodb.net" in uri_lower and "tlscafile=" not in uri_lower:
                client_kwargs["tlsCAFile"] = certifi.where()

            self.client = AsyncIOMotorClient(settings.mongodb_uri, **client_kwargs)
            await self.client.admin.command("ping")
            self.database = self.client[settings.mongodb_db_name]
            await self._ensure_indexes()
            self.connected = True
            self.last_error = None
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self.connected = False
            self.last_error = str(exc)
            self.database = None

    async def disconnect(self) -> None:
        if self.client is not None:
            self.client.close()
        self.client = None
        self.database = None
        self.connected = False

    async def _ensure_indexes(self) -> None:
        if self.database is None:
            return
        await self.database["users"].create_index("email", unique=True)
        await self.database["foods"].create_index("slug", unique=True)
        await self.database["meal_logs"].create_index([("user_id", 1), ("created_at", -1)])

    def get_collection(self, name: str) -> AsyncIOMotorCollection:
        if self.database is None:
            raise RuntimeError("MongoDB is not connected.")
        return self.database[name]

    async def ping(self) -> bool:
        if self.client is None:
            return False
        try:
            await self.client.admin.command("ping")
            self.connected = True
            self.last_error = None
            return True
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self.connected = False
            self.last_error = str(exc)
            return False


mongo_manager = MongoManager()


async def get_database() -> AsyncIterator[AsyncIOMotorDatabase]:
    if mongo_manager.database is None:
        raise RuntimeError("MongoDB is not connected.")
    yield mongo_manager.database


def get_collection(name: str) -> AsyncIOMotorCollection[Any]:
    return mongo_manager.get_collection(name)
