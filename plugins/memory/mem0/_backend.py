"""Backend abstraction for Mem0 Platform and OSS modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Mem0Backend(ABC):
    """Unified interface over Platform (MemoryClient) and OSS (Memory) backends."""

    @abstractmethod
    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = True) -> list[dict]:
        ...

    @abstractmethod
    def get_all(self, *, filters: dict, page: int = 1, page_size: int = 100) -> dict:
        ...

    @abstractmethod
    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False) -> dict:
        ...

    @abstractmethod
    def update(self, memory_id: str, text: str) -> dict:
        ...

    @abstractmethod
    def delete(self, memory_id: str) -> dict:
        ...


def _unwrap_results(response: Any) -> list:
    """Normalize API response — extract results list from dict or pass through."""
    if isinstance(response, dict):
        return response.get("results", [])
    if isinstance(response, list):
        return response
    return []


class PlatformBackend(Mem0Backend):
    """Wraps mem0.MemoryClient for Mem0 Platform (cloud API)."""

    def __init__(self, api_key: str):
        from mem0 import MemoryClient
        self._client = MemoryClient(api_key=api_key)

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = True) -> list[dict]:
        response = self._client.search(query, filters=filters, top_k=top_k, rerank=rerank)
        return _unwrap_results(response)

    def get_all(self, *, filters: dict, page: int = 1, page_size: int = 100) -> dict:
        response = self._client.get_all(filters=filters, page=page, page_size=page_size)
        results = response.get("results", []) if isinstance(response, dict) else response
        count = response.get("count", len(results)) if isinstance(response, dict) else len(results)
        return {"results": results, "count": count}

    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False) -> dict:
        return self._client.add(messages, user_id=user_id, agent_id=agent_id, infer=infer)

    def update(self, memory_id: str, text: str) -> dict:
        self._client.update(memory_id=memory_id, text=text)
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id: str) -> dict:
        self._client.delete(memory_id=memory_id)
        return {"result": "Memory deleted.", "memory_id": memory_id}


class OSSBackend(Mem0Backend):
    """Wraps mem0.Memory for self-hosted (OSS) mode."""

    def __init__(self, oss_config: dict):
        from mem0 import Memory

        vector_store = dict(oss_config["vector_store"])
        vs_config = dict(vector_store.get("config", {}))

        embedder_config = oss_config.get("embedder", {}).get("config", {})
        dims = embedder_config.get("embedding_dims")
        if not dims:
            from ._oss_providers import KNOWN_DIMS
            model = embedder_config.get("model", "")
            dims = KNOWN_DIMS.get(model)
        if dims and "embedding_model_dims" not in vs_config:
            vs_config["embedding_model_dims"] = dims

        vector_store["config"] = vs_config

        config = {
            "vector_store": vector_store,
            "llm": oss_config["llm"],
            "embedder": oss_config["embedder"],
            "version": "v1.1",
        }
        self._memory = Memory.from_config(config)

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = True) -> list[dict]:
        response = self._memory.search(query, filters=filters, top_k=top_k)
        return _unwrap_results(response)

    def get_all(self, *, filters: dict, page: int = 1, page_size: int = 100) -> dict:
        response = self._memory.get_all(filters=filters)
        results = _unwrap_results(response)
        return {"results": results, "count": len(results)}

    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False) -> dict:
        return self._memory.add(messages, user_id=user_id, agent_id=agent_id, infer=infer)

    def update(self, memory_id: str, text: str) -> dict:
        self._memory.update(memory_id, data=text)
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id: str) -> dict:
        self._memory.delete(memory_id)
        return {"result": "Memory deleted.", "memory_id": memory_id}

    def close(self):
        try:
            telemetry = getattr(self._memory, "telemetry", None)
            if telemetry and hasattr(telemetry, "posthog"):
                try:
                    telemetry.posthog.shutdown()
                except Exception:
                    pass
            if hasattr(self._memory, "close"):
                self._memory.close()
            vs = getattr(self._memory, "vector_store", None)
            if vs and hasattr(vs, "close"):
                vs.close()
            client = getattr(vs, "client", None)
            if client and hasattr(client, "close"):
                client.close()
        except Exception:
            pass
