# Mem0 Memory Provider

Server-side LLM fact extraction with semantic search and hybrid multi-signal retrieval via the Mem0 Platform v3 API.

## Requirements

- `pip install mem0ai`
- Mem0 API key from [app.mem0.ai](https://app.mem0.ai)

## Setup

```bash
hermes memory setup    # select "mem0"
```

Or manually:
```bash
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

## Config

Config file: `$HERMES_HOME/mem0.json`

| Key | Default | Description |
|-----|---------|-------------|
| `user_id` | `hermes-user` | User identifier on Mem0 |
| `agent_id` | `hermes` | Agent identifier |
| `rerank` | `false` | Enable reranking for precision (adds latency) |

## Tools

| Tool | Description |
|------|-------------|
| `mem0_list` | List all stored memories (paginated) |
| `mem0_search` | Semantic search by meaning |
| `mem0_add` | Store a fact verbatim (no LLM extraction) |
| `mem0_update` | Update a memory's text by ID |
| `mem0_delete` | Delete a memory by ID |
