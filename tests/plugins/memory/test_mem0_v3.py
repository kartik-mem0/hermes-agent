"""Tests for Mem0 v3 API — new tool names, paginated responses, update/delete tools."""

import json
import pytest

from plugins.memory.mem0 import Mem0MemoryProvider


class FakeClientV3:
    """Fake Mem0 client returning v3-style responses."""

    def __init__(self, search_results=None, all_results=None):
        self._search_results = search_results or {"results": []}
        self._all_results = all_results or {
            "count": 0, "next": None, "previous": None, "results": []
        }
        self.captured_search = {}
        self.captured_get_all = {}
        self.captured_add = []
        self.captured_update = []
        self.captured_delete = []

    def search(self, query, **kwargs):
        self.captured_search = {"query": query, **kwargs}
        return self._search_results

    def get_all(self, **kwargs):
        self.captured_get_all = kwargs
        return self._all_results

    def add(self, messages, **kwargs):
        self.captured_add.append({"messages": messages, **kwargs})
        return {"status": "PENDING", "event_id": "evt-test-123"}

    def update(self, **kwargs):
        self.captured_update.append(kwargs)
        return {"id": kwargs.get("memory_id"), "text": kwargs.get("text"),
                "updated_at": "2026-04-25T00:00:00Z"}

    def delete(self, **kwargs):
        self.captured_delete.append(kwargs)


class TestMem0V3Tools:
    """Test v3 tool names and response handling."""

    def _make_provider(self, monkeypatch, client):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        monkeypatch.setattr(provider, "_get_client", lambda: client)
        return provider

    def test_list_returns_paginated_with_ids(self, monkeypatch):
        client = FakeClientV3(all_results={
            "count": 2, "next": None, "previous": None,
            "results": [
                {"id": "mem-1", "memory": "alpha"},
                {"id": "mem-2", "memory": "beta"},
            ]
        })
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_list", {}))
        assert result["count"] == 2
        assert result["results"][0]["id"] == "mem-1"
        assert result["results"][0]["memory"] == "alpha"

    def test_list_pagination_params(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        provider.handle_tool_call("mem0_list", {"page": 2, "page_size": 50})
        assert client.captured_get_all["page"] == 2
        assert client.captured_get_all["page_size"] == 50

    def test_list_empty(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_list", {}))
        assert result["result"] == "No memories stored yet."

    def test_search_returns_ids(self, monkeypatch):
        client = FakeClientV3(search_results={
            "results": [{"id": "mem-1", "memory": "foo", "score": 0.9}]
        })
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_search", {"query": "test"}))
        assert result["results"][0]["id"] == "mem-1"

    def test_search_rerank_default_false(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        provider.handle_tool_call("mem0_search", {"query": "test"})
        assert client.captured_search["rerank"] is False

    def test_search_rerank_override(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        provider.handle_tool_call("mem0_search", {"query": "test", "rerank": True})
        assert client.captured_search["rerank"] is True

    def test_search_uses_filters(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        provider.handle_tool_call("mem0_search", {"query": "hello", "top_k": 3})
        assert client.captured_search["filters"] == {"user_id": "u123"}
        assert client.captured_search["top_k"] == 3

    def test_add_uses_content_param(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_add", {"content": "user likes dark mode"}))
        assert len(client.captured_add) == 1
        call = client.captured_add[0]
        assert call["infer"] is False
        assert call["user_id"] == "u123"
        assert call["agent_id"] == "hermes"
        assert "event_id" in result

    def test_add_returns_event_id(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_add", {"content": "test"}))
        assert result["event_id"] == "evt-test-123"

    def test_add_missing_content(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_add", {}))
        assert "error" in result

    def test_unwrap_results_edge_cases(self):
        """_unwrap_results handles all response shapes."""
        assert Mem0MemoryProvider._unwrap_results({"results": [1, 2]}) == [1, 2]
        assert Mem0MemoryProvider._unwrap_results([3, 4]) == [3, 4]
        assert Mem0MemoryProvider._unwrap_results({}) == []
        assert Mem0MemoryProvider._unwrap_results(None) == []
        assert Mem0MemoryProvider._unwrap_results("unexpected") == []


class TestMem0UpdateDelete:

    def _make_provider(self, monkeypatch, client):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        monkeypatch.setattr(provider, "_get_client", lambda: client)
        return provider

    def test_update_calls_sdk(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": "mem-1", "text": "updated fact"}
        ))
        assert client.captured_update[0] == {"memory_id": "mem-1", "text": "updated fact"}
        assert result["result"] == "Memory updated."
        assert result["memory_id"] == "mem-1"

    def test_update_missing_memory_id(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_update", {"text": "no id"}))
        assert "error" in result

    def test_update_missing_text(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_update", {"memory_id": "mem-1"}))
        assert "error" in result

    def test_delete_calls_sdk(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call(
            "mem0_delete", {"memory_id": "mem-1"}
        ))
        assert client.captured_delete[0] == {"memory_id": "mem-1"}
        assert result["result"] == "Memory deleted."

    def test_delete_missing_memory_id(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_delete", {}))
        assert "error" in result


class TestMem0ErrorHandling:

    def _make_provider(self, monkeypatch, client):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        monkeypatch.setattr(provider, "_get_client", lambda: client)
        return provider

    def test_update_404_no_circuit_breaker(self, monkeypatch):
        client = FakeClientV3()
        client.update = lambda **kw: (_ for _ in ()).throw(Exception("404 Not Found"))
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": "bad-id", "text": "x"}
        ))
        assert "error" in result
        assert provider._consecutive_failures == 0

    def test_delete_404_no_circuit_breaker(self, monkeypatch):
        client = FakeClientV3()
        client.delete = lambda **kw: (_ for _ in ()).throw(Exception("404 not found"))
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call(
            "mem0_delete", {"memory_id": "bad-id"}
        ))
        assert "error" in result
        assert provider._consecutive_failures == 0

    def test_update_validation_error_no_circuit_breaker(self, monkeypatch):
        """ValidationError (bad UUID format) should not trip circuit breaker."""
        class ValidationError(Exception):
            pass
        client = FakeClientV3()
        client.update = lambda **kw: (_ for _ in ()).throw(
            ValidationError('{"error":"memory_id should be a valid UUID"}')
        )
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": "not-a-uuid", "text": "x"}
        ))
        assert "error" in result
        assert provider._consecutive_failures == 0

    def test_delete_validation_error_no_circuit_breaker(self, monkeypatch):
        class ValidationError(Exception):
            pass
        client = FakeClientV3()
        client.delete = lambda **kw: (_ for _ in ()).throw(
            ValidationError('{"error":"memory_id should be a valid UUID"}')
        )
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call(
            "mem0_delete", {"memory_id": "not-a-uuid"}
        ))
        assert "error" in result
        assert provider._consecutive_failures == 0

    def test_update_5xx_trips_circuit_breaker(self, monkeypatch):
        client = FakeClientV3()
        client.update = lambda **kw: (_ for _ in ()).throw(Exception("500 Internal Server Error"))
        provider = self._make_provider(monkeypatch, client)
        provider.handle_tool_call("mem0_update", {"memory_id": "mem-1", "text": "x"})
        assert provider._consecutive_failures == 1


class TestMem0V3Internal:

    def _make_provider(self, monkeypatch, client):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        monkeypatch.setattr(provider, "_get_client", lambda: client)
        return provider

    def test_sync_turn_explicit_kwargs(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        provider.sync_turn("user said", "assistant replied", session_id="s1")
        provider._sync_thread.join(timeout=2)
        assert len(client.captured_add) == 1
        call = client.captured_add[0]
        assert call["user_id"] == "u123"
        assert call["agent_id"] == "hermes"
        assert "infer" not in call

    def test_prefetch_uses_config_rerank(self, monkeypatch):
        client = FakeClientV3(search_results={
            "results": [{"memory": "test fact"}]
        })
        provider = self._make_provider(monkeypatch, client)
        provider._rerank = False
        provider.queue_prefetch("test query")
        provider._prefetch_thread.join(timeout=2)
        assert client.captured_search["rerank"] is False

    def test_old_tool_names_return_unknown(self, monkeypatch):
        client = FakeClientV3()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_profile", {}))
        assert "error" in result
        result = json.loads(provider.handle_tool_call("mem0_conclude", {}))
        assert "error" in result


class TestMem0V3Config:

    def test_rerank_default_false(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = Mem0MemoryProvider()
        provider.initialize("test")
        assert provider._rerank is False

    def test_tool_schemas_five_tools(self):
        provider = Mem0MemoryProvider()
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert names == ["mem0_list", "mem0_search", "mem0_add", "mem0_update", "mem0_delete"]

    def test_system_prompt_new_tool_names(self):
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        block = provider.system_prompt_block()
        assert "mem0_search" in block
        assert "mem0_add" in block
        assert "mem0_list" in block
        assert "mem0_update" in block
        assert "mem0_delete" in block
        assert "mem0_profile" not in block
        assert "mem0_conclude" not in block
