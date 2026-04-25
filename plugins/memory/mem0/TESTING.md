# Testing the Mem0 v3 Plugin Inside Hermes

This guide walks through testing the upgraded Mem0 plugin end-to-end inside Hermes Agent — from setup to verifying every tool works in a live session.

## Prerequisites

- Hermes Agent installed (editable/dev mode from this repo)
- Mem0 API key from [app.mem0.ai](https://app.mem0.ai)
- Python 3.11+

## Step 1: Install the v3 SDK

```bash
pip install "mem0ai>=2.0.0,<3"
```

Verify version:

```bash
python3 -c "import mem0; print(mem0.__version__)"
```

Should print `2.x.x`.

## Step 2: Configure Mem0 as the Memory Provider

### Option A: Interactive setup

```bash
hermes memory setup
```

Select `mem0` from the list. It will prompt for:
- **API key** — paste your key from app.mem0.ai
- **User ID** — use a test ID like `hermes-test-user` (keeps test data isolated from real data)
- **Agent ID** — use `hermes` (default)
- **Rerank** — choose `false` for speed or `true` for precision

### Option B: Manual setup

```bash
# Set the provider in config
hermes config set memory.provider mem0

# Add API key to environment
echo "MEM0_API_KEY=m0-your-key-here" >> ~/.hermes/.env

# Create mem0.json config
cat > ~/.hermes/mem0.json << 'EOF'
{
  "user_id": "hermes-test-user",
  "agent_id": "hermes",
  "rerank": false
}
EOF
```

## Step 3: Verify Provider Status

```bash
hermes memory status
```

Expected output:

```
Memory status
────────────────────────────────────
  Built-in:  always active
  Provider:  mem0

  Plugin:    installed ✓
  Status:    available ✓

  Installed plugins:
    • mem0  (Mem0 — server-side LLM fact extraction...)  ← active
```

If it shows `not available ✗`, check that `MEM0_API_KEY` is set in `~/.hermes/.env`.

## Step 4: Run Unit Tests

These use a fake client — no API key needed. Run from the repo root:

```bash
pytest tests/plugins/memory/test_mem0_v3.py -v -p no:xdist -o "addopts="
```

Expected: **25/25 passed**.

## Step 5: Test Tools in a Live Hermes Session

Start Hermes:

```bash
hermes
```

Once in the session, you should see `Mem0 Memory` mentioned in the system context. Test each tool by asking the agent:

### 5a. Test `mem0_add`

```
Remember that I prefer dark mode in all my editors.
```

The agent should use `mem0_add` to store this fact. Look for tool call output showing:
- Tool: `mem0_add`
- Result: `"Fact queued for storage."` with an `event_id`

### 5b. Test `mem0_list`

```
Show me all my stored memories.
```

The agent should use `mem0_list`. Look for:
- Tool: `mem0_list`
- Result: A list of memories with `id` and `memory` fields
- The dark mode preference from 5a should appear (may take a few seconds for async processing)

### 5c. Test `mem0_search`

```
Search my memories for anything about editors.
```

The agent should use `mem0_search`. Look for:
- Tool: `mem0_search`
- Result: Memories with `id`, `memory`, and `score` fields
- The dark mode preference should appear with a relevance score

### 5d. Test `mem0_update`

```
Update that dark mode memory — I actually prefer light mode now.
```

The agent should:
1. Use `mem0_search` or `mem0_list` to find the memory ID
2. Use `mem0_update` with that ID and the new text

Look for:
- Tool: `mem0_update`
- Result: `"Memory updated."` with the `memory_id`

### 5e. Test `mem0_delete`

```
Delete the memory about my editor theme preference.
```

The agent should:
1. Find the memory ID via search or list
2. Use `mem0_delete` with that ID

Look for:
- Tool: `mem0_delete`
- Result: `"Memory deleted."` with the `memory_id`

### 5f. Verify deletion

```
List all my memories again.
```

The deleted memory should no longer appear.

## Step 6: Test Auto-Sync (sync_turn)

This happens automatically every turn — Hermes sends the conversation to Mem0 for fact extraction (with `infer=True`).

In a Hermes session, have a natural conversation:

```
I'm working on a Rust project at ~/code/my-api using Axum.
```

Then in a **new session** (exit and restart Hermes):

```bash
hermes
```

Ask:

```
What do you remember about my projects?
```

The agent should find the Rust/Axum fact via `mem0_search` or prefetch — this confirms sync_turn is working and persisting memories across sessions.

## Step 7: Test Prefetch

Prefetch runs automatically before each turn. To verify:

1. Start Hermes with debug logging:

```bash
hermes --log-level debug
```

2. Send any message. In the logs (`~/.hermes/logs/`), look for:

```
mem0-prefetch - Mem0 prefetch...
```

This confirms background memory retrieval is running before each turn.

## Step 8: Test Rerank

In a Hermes session:

```
Search my memories for project details, use reranking for precision.
```

The agent should call `mem0_search` with `rerank: true`. You can also test by setting rerank in config:

```bash
# Edit mem0.json
cat ~/.hermes/mem0.json
# Change "rerank": true
```

Restart Hermes and verify prefetch uses reranking (visible in debug logs).

## Step 9: Test Error Handling

In a Hermes session, ask the agent to deliberately trigger errors:

```
Try to update a memory with ID "nonexistent-fake-id-12345" to say "test".
```

The agent should use `mem0_update` and get back an error like `"Memory not found: nonexistent-fake-id-12345"`. It should handle this gracefully — no crashes, no circuit breaker trips.

## Step 10: Verify Old Tool Names Are Gone

If you have any custom prompts or scripts referencing `mem0_profile` or `mem0_conclude`, they will now get `"Unknown tool"` errors. Search your config:

```bash
grep -r "mem0_profile\|mem0_conclude" ~/.hermes/ 2>/dev/null
```

Update any references to use the new names:
- `mem0_profile` → `mem0_list`
- `mem0_conclude` → `mem0_add`

## Cleanup

After testing, clean up test memories:

```bash
hermes memory off
```

Or delete test user's data entirely via Python:

```python
from mem0 import MemoryClient
import os

client = MemoryClient(api_key=os.environ["MEM0_API_KEY"])
client.delete_all(user_id="hermes-test-user")
print("Test memories cleaned up")
```

Then switch back to your real user ID in `~/.hermes/mem0.json` or re-run `hermes memory setup`.

## Quick Reference: What Changed from v2

| v2 | v3 |
|----|-----|
| `mem0_profile` | `mem0_list` (now paginated, returns IDs) |
| `mem0_conclude` | `mem0_add` (param: `conclusion` → `content`) |
| `mem0_search` with `rerank=true` default | `mem0_search` with `rerank=false` default |
| — | `mem0_update` (new) |
| — | `mem0_delete` (new) |
| `add()` returns memory objects | `add()` returns `PENDING` + `event_id` |
| `get_all()` returns flat list | `get_all()` returns paginated envelope |
