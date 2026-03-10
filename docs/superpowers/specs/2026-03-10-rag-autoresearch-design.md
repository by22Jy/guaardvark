# RAG Autoresearch — Design Specification

**Date:** 2026-03-10
**Status:** Approved
**Inspired by:** [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — autonomous AI experiment loops

## Summary

An autonomous RAG optimization system that runs experiments while the system is idle, measuring and improving retrieval quality without human intervention. Each Guaardvark instance (Nephew) optimizes for its own indexed corpus. Winning configurations are shared across the family via the interconnector. Uncle Claude acts as an optional research director.

**Key properties:**
- Offline-first: entire loop runs on local Ollama, zero external API dependency
- Self-tuning: auto-generates eval sets from whatever the user indexes
- Resource-respectful: only runs when system is idle, pauses instantly on user activity
- Family-aware: shares discoveries across instances via interconnector
- HIPAA-compatible: no data leaves the network

## Motivation

Karpathy's autoresearch demonstrated that an AI agent given a training script, a fixed time budget, and a single metric can autonomously run 100+ experiments overnight, finding improvements a human researcher missed over two decades. The same pattern applies to RAG pipeline optimization — there are dozens of configurable parameters (chunk_size, top_k, similarity_threshold, chunking strategies) and no one-size-fits-all configuration. Every corpus is different.

Currently, Guaardvark users must manually tune RAG parameters or accept defaults. This feature makes every Guaardvark instance self-optimizing.

## Architecture

### Core Loop (mirrors Karpathy's structure)

```
Karpathy's autoresearch          Guaardvark RAG Autoresearch
--------------------------       --------------------------------
prepare.py (immutable eval)  ->  RAGEvalHarness (auto-generated Q&A + LLM-as-judge)
train.py (agent edits this)  ->  rag_experiment_config.json (parameter search space)
program.md (human edits)     ->  research_program.md (research directives)
results.tsv (experiment log) ->  ExperimentRun DB table (queryable history)
```

### One Experiment Cycle

```
1. Load current best config (rag_experiment_config.json)
2. Agent reads experiment history + research_program.md
3. Agent proposes ONE parameter change with hypothesis
4. Apply config change temporarily
5. Run eval harness:
   a. Query-time change? -> Re-run eval queries against existing index
   b. Index-time change? -> Re-index shadow corpus, then run eval queries
6. LLM-as-judge scores each response -> composite score
7. Compare to baseline:
   - Better -> keep config, update baseline, log as "keep"
   - Worse/equal -> revert config, log as "discard"
8. Broadcast result via interconnector (if family connected)
9. Repeat (or yield if user becomes active)
```

Eval queries run through the **real** retrieval pipeline — the same code path users hit. Improvements transfer directly to user experience.

### Config Injection Path (prerequisite plumbing)

Currently, `search_with_llamaindex()` in `indexing_service.py` and `_retrieve_rag_context()` in `unified_chat_engine.py` have hardcoded retrieval parameters (e.g., `max_chunks=3`). The autoresearch loop needs to inject experiment configs into this path.

**Solution: Thread-local experiment context.**

```python
# New: backend/utils/experiment_context.py
import threading

_experiment_config = threading.local()

def set_experiment_config(config: dict):
    """Set experiment params for current thread (used during eval runs)."""
    _experiment_config.params = config

def get_experiment_config() -> dict | None:
    """Get experiment params, or None if not in an experiment."""
    return getattr(_experiment_config, 'params', None)

def clear_experiment_config():
    _experiment_config.params = None
```

The retrieval path checks for an active experiment config:
```python
# In search_with_llamaindex():
from backend.utils.experiment_context import get_experiment_config

def search_with_llamaindex(query, max_chunks=3, project_id=None):
    exp = get_experiment_config()
    if exp:
        max_chunks = exp.get('context_window_chunks', max_chunks)
        top_k = exp.get('top_k', 5)
        # ... apply other Phase 1 params
    # ... rest of existing function
```

This approach:
- Zero impact on normal user queries (experiment context is None outside eval runs)
- No function signature changes needed on existing code
- Thread-safe (Celery workers run eval in their own threads)
- The experiment orchestrator wraps each eval run with `set_experiment_config()` / `clear_experiment_config()`

**For Phase 1 parameters that need `hybrid_rag_pipeline.py`:** The existing `HybridRAGPipeline` class supports `query_expansion`, `reranking`, and `hybrid_search_alpha` but is not currently wired into `unified_chat_engine.py`. The config injection path will route through `HybridRAGPipeline` when any of these features are enabled in the experiment config, and through the standard `search_with_llamaindex` path otherwise.

### Service Architecture

```
                    +-------------------------+
                    |   Celery Beat /          |
                    |   Idle Detector /         |
                    |   Manual Trigger          |
                    +-----------+--------------+
                                |
                    +-----------v--------------+
                    | rag_autoresearch_         |
                    | service.py                |
                    | (Orchestrator)            |
                    +---+------+------+--------+
                        |      |      |
           +------------+      |      +-----------+
           v                   v                   v
+------------------+ +----------------+ +---------------------+
| rag_experiment_  | | rag_eval_      | | indexing_service.py  |
| agent.py         | | harness.py     | | (existing - shadow   |
| (Hypothesis +   | | (Generate eval | |  re-index)           |
|  config mutation)| |  pairs, judge) | +---------------------+
+------------------+ +----------------+
                            |
                    +-------v------------------+
                    | unified_chat_engine.py   |
                    | (existing - runs eval    |
                    |  queries through real     |
                    |  RAG pipeline)            |
                    +--------------------------+
```

## Eval Harness (the "prepare.py")

### Auto-Generated Eval Set

When documents are indexed (or on-demand), the system generates eval pairs:

```
For each sampled indexed document/chunk:
  -> Local LLM prompt: "Given this text, generate a factual question
     that someone would ask, and the correct answer based ONLY on this text."
  -> Output: { question, expected_answer, source_doc_id, source_chunk }
```

- Target: 50-200 eval pairs per corpus (scales with corpus size)
- **Minimum corpus threshold:** Autoresearch requires at least 10 indexed documents to generate meaningful eval pairs. Below this threshold, the feature is disabled with a dashboard notification: "Index more documents to enable auto-optimization."
- Stored in `EvalPair` DB table
- Immutable during an experiment run (locked by a `eval_generation_id` field)
- Stratified across corpus types (code, knowledge, client data)
- **Staleness detection:** Each eval pair stores a `source_chunk_hash`. On each experiment cycle start, 10% of pairs are spot-checked against current index. If >20% of sampled hashes have changed, eval regeneration is triggered before the next experiment (not mid-run).
- **Regeneration triggers:** (1) Staleness check fails, (2) new indexing event adds >10% more documents, (3) manual regeneration via API, (4) corpus type distribution changes significantly

### LLM-as-Judge Scoring

For each eval pair:
```
1. Query RAG pipeline with eval_pair.question using current config
2. Get response + retrieved_chunks
3. LLM-as-judge scores three dimensions (1-5 each, higher = better):
   - Relevance: "Are the retrieved chunks relevant to the question?"
   - Grounding: "Is the answer supported by the retrieved chunks?"
   - Completeness: "Does the answer fully address the question?"
4. Composite quality score = weighted average of dimensions (higher = better, range 1.0-5.0)
```

**Score convention: higher is better.** Unlike Karpathy's `val_bpb` (a loss where lower = better), our metric is a quality score where 5.0 is perfect. This is more intuitive for users viewing the dashboard. A positive `delta` means improvement; negative means regression. The experiment cycle keeps changes where `composite_score > baseline_score`.

**Judge model considerations:**
- Uses the same local Ollama LLM but with `temperature=0.0` for deterministic scoring
- Judge prompt is structured to return JSON `{"relevance": N, "grounding": N, "completeness": N}` for reliable parsing
- Minimum recommended model size for judging: 7B parameters (smaller models produce unreliable scores)
- If available, a separate model can be configured for judging via `rag_autoresearch_judge_model` setting
- Calibration: first eval run establishes baseline; relative improvements matter more than absolute scores

The composite score is the single metric — Karpathy's `val_bpb` equivalent.

### Shadow Eval Corpus

For index-time experiments (Phase 2), a lightweight parallel index is maintained:
- Subset of ~100 documents sampled from full index (stratified by corpus type)
- Stored in a separate directory: `data/autoresearch/shadow_index/` with its own `StorageContext`
- Uses its own `_shadow_index_lock` (separate from the main `_index_operation_lock` to avoid blocking user indexing)
- Fast to re-index (seconds, not minutes) due to small corpus size
- Eval pairs for Phase 2 are generated exclusively from this subset
- Winning Phase 2 configs are applied to the full index as a background Celery task (non-blocking)
- Shadow corpus is refreshed when the main index changes significantly (same staleness triggers as eval pairs)
- Disk space estimate: shadow index is ~1/100th the size of main index. Guard: skip Phase 2 if estimated shadow size > available disk * 0.5%, or if available disk < 1GB

## Search Space (the "train.py")

### Phase 1 — Query-Time Parameters (no re-indexing needed)

| Parameter | Default | Range | Description |
|---|---|---|---|
| `top_k` | 5 | 1-20 | Number of chunks retrieved from vector store (passed to retriever's `similarity_top_k`) |
| `dedup_threshold` | 0.85 | 0.5-0.98 | Post-retrieval deduplication cutoff (existing `deduplicate_chunks()` threshold — removes near-duplicate chunks after retrieval) |
| `context_window_chunks` | 3 | 1-10 | How many of the retrieved chunks are actually included in the LLM context (top_k retrieves candidates, this selects the best N). Currently hardcoded as `max_chunks=3` in `search_with_llamaindex()` |
| `reranking_enabled` | false | bool | Re-rank retrieved chunks by query relevance before selecting context_window_chunks (uses `HybridRAGPipeline.rerank()`) |
| `query_expansion` | false | bool | Expand query with LLM-generated synonyms/rephrasing before retrieval (uses `HybridRAGPipeline.expand_query()`) |
| `hybrid_search_alpha` | 0.0 | 0.0-1.0 | Blend between vector similarity (0.0) and BM25 keyword search (1.0). 0.0 = pure vector (current default). Requires `HybridRAGPipeline` routing |

**Note on parameter interactions:** The "ONE change per experiment" rule applies to initial exploration. After Phase 1 single-parameter exploration plateaus, the agent may propose paired changes (e.g., `top_k=10` + `context_window_chunks=5`) as a single experiment, logged with both parameters in the hypothesis. The research program guides when this is appropriate.

### Phase 2 — Index-Time Parameters (shadow corpus re-index)

| Parameter | Default | Range | Description |
|---|---|---|---|
| `chunk_size` | 1000 | 200-3000 | Tokens per chunk |
| `chunk_overlap` | 200 | 0-500 | Overlap between chunks |
| `use_semantic_splitting` | false | bool | Semantic boundary splitting |
| `use_hierarchical_splitting` | false | bool | Parent-child chunks |
| `extract_entities` | false | bool | Entity extraction |
| `preserve_structure` | false | bool | Maintain doc structure |

### Phase 3 — Model-Level (full re-index, user opt-in required)

| Parameter | Default | Range | Description |
|---|---|---|---|
| `embedding_model` | current | available list | Switch embedding model |

### Phase Transition

If 10 consecutive experiments in current phase are "discard" (no improvement), advance to next phase. Config stored in `rag_experiment_config.json`:

```json
{
  "version": 1,
  "baseline_score": 0.0,
  "params": { "top_k": 5, "similarity_threshold": 0.85, ... },
  "phase": 1,
  "phase_plateau_count": 0
}
```

## Research Program (the "program.md")

Lives at `data/research_program.md`. Ships with sensible defaults:

```markdown
# RAG Autoresearch Program

## Your Role
You are an autonomous RAG optimization researcher. You read experiment
history, form hypotheses, propose ONE parameter change per cycle, and
evaluate results. You work indefinitely without human intervention.

## Rules
- Modify only parameters in rag_experiment_config.json
- ONE change per experiment (isolate variables)
- If 3 consecutive experiments crash, revert to last known good config
- Prefer simplicity: if two configs score equally, keep the simpler one
- Log your reasoning in the experiment description

## Strategy
1. Start with Phase 1 (query-time) parameters
2. Try large changes first to find the ballpark, then fine-tune
3. When you see a pattern, push it further until it stops helping
4. Different corpus types may want different params
5. If stuck, try combining two previous near-misses
```

**Uncle Claude updates:** When available, Uncle Claude reviews experiment logs and appends insights to the research program (e.g., "cap top_k exploration at 12 — diminishing returns above that").

**Family sharing:** Winning insights appended under `## Family Insights` section via interconnector broadcasts.

## Database Schema

### ExperimentRun

| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | Unique experiment ID |
| `run_tag` | String | Group tag (e.g., "mar10-nightly") |
| `phase` | Integer | 1=query, 2=index, 3=model |
| `parameter_changed` | String | Which knob was turned (or comma-separated for paired changes) |
| `old_value` | String | Previous value(s) |
| `new_value` | String | Attempted value(s) |
| `hypothesis` | Text | Agent's reasoning |
| `composite_score` | Float | Eval result (higher=better, range 1.0-5.0) |
| `baseline_score` | Float | Score before experiment |
| `delta` | Float | composite - baseline (positive = improvement) |
| `status` | Enum | keep, discard, crash |
| `eval_details` | JSON | Per-question scores {relevance, grounding, completeness} |
| `duration_seconds` | Float | Wall clock time |
| `node_id` | String(36), nullable | Which Nephew ran this. Nullable for standalone (non-family) instances. FK to `interconnector_nodes.node_id` when interconnector is active, otherwise stores a local machine identifier |
| `created_at` | DateTime | Timestamp |

### EvalPair

| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | |
| `eval_generation_id` | String | Groups pairs from the same generation run (immutability lock) |
| `question` | Text | Auto-generated question |
| `expected_answer` | Text | Expected answer |
| `source_doc_id` | Integer(FK) | FK to `documents.id` |
| `source_chunk_hash` | String(64) | SHA256 of source chunk text (staleness detection) |
| `corpus_type` | Enum | code, knowledge, client |
| `quality_score` | Float, nullable | Organic feedback over time (future: thumbs up/down) |
| `created_at` | DateTime | |

### ResearchConfig

| Column | Type | Description |
|---|---|---|
| `id` | UUID (PK) | |
| `params` | JSON | Full parameter snapshot |
| `composite_score` | Float | Score achieved (higher=better) |
| `is_active` | Boolean | Currently applied |
| `promoted_at` | DateTime | When this became active |
| `source` | Enum | local, family_broadcast, uncle_directive |

### Relationships to Existing Tables

- `ExperimentRun.node_id` -> `InterconnectorNode.node_id` (nullable FK, null on standalone instances)
- `EvalPair.source_doc_id` -> `Document.id`
- **InterconnectorLearning integration:** Winning configs are broadcast as `InterconnectorLearning` records. The config JSON is serialized into the existing `description` field (Text) as a structured format: `"[AUTORESEARCH] param=value, score=X.XX, delta=+Y.YY\n{json_config}"`. The `code_diff` field stores the full `rag_experiment_config.json` diff. `learning_type` set to `"rag_optimization"`. `source_node_id` matches the local node's ID.
- **SelfImprovementRun integration:** Experiment summaries written to `SelfImprovementRun` with `trigger` field (not `trigger_type`) set to `"autoresearch"`. Note: the existing trigger values are `scheduled, reactive, directed, family_learning` — the `"autoresearch"` value must be added as a valid option in the model's comment/validation.
- **Settings storage:** Autoresearch settings (`idle_minutes`, `auto_enabled`, `max_experiments`, `phase_limit`, `judge_model`) stored in the existing `Setting` DB model (key-value pairs, same pattern as `active_embedding_model`).

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/autoresearch/start` | Start experiment loop |
| `POST` | `/api/autoresearch/stop` | Graceful pause |
| `GET` | `/api/autoresearch/status` | Current state + progress |
| `GET` | `/api/autoresearch/history` | Paginated experiment log |
| `GET` | `/api/autoresearch/config` | Current best config |
| `POST` | `/api/autoresearch/config/reset` | Reset to defaults |
| `GET` | `/api/autoresearch/eval-pairs` | View eval set |
| `POST` | `/api/autoresearch/eval-pairs/regenerate` | Force regenerate |
| `GET` | `/api/autoresearch/insights` | Family-aggregated insights |
| `GET` | `/api/autoresearch/settings` | Autoresearch settings |
| `PUT` | `/api/autoresearch/settings` | Update settings |

### Socket.IO Events

| Event | Direction | Payload |
|---|---|---|
| `autoresearch:experiment_start` | Server->Client | `{experiment_id, parameter, hypothesis}` |
| `autoresearch:experiment_complete` | Server->Client | `{experiment_id, status, score, delta}` |
| `autoresearch:phase_change` | Server->Client | `{old_phase, new_phase}` |
| `autoresearch:family_broadcast` | Server->Client | `{node_name, parameter, improvement}` |

## Idle Detection & Triggers

### Activity Tracking

Track last user activity timestamp (updated on every non-autoresearch API request and Socket.IO chat event).

### Settings

| Setting | Default | Range | Description |
|---|---|---|---|
| `rag_autoresearch_idle_minutes` | 10 | 5-120 | Minutes of inactivity before loop starts |
| `rag_autoresearch_auto_enabled` | true | bool | Enable idle auto-trigger |
| `rag_autoresearch_max_experiments` | 0 | 0=unlimited | Cap per session |
| `rag_autoresearch_phase_limit` | 2 | 1-3 | Max phase allowed |

### Three Trigger Paths

1. **Idle trigger** — Celery Beat checks every 60s, starts if idle > threshold
2. **Event trigger** — Indexing completion marks eval set as stale, next idle cycle regenerates and runs
3. **Manual trigger** — User clicks "Optimize Now" from dashboard

**Redis/Celery fallback:** If Celery Beat is unavailable (Redis down), idle detection cannot fire. The autoresearch status endpoint (`GET /api/autoresearch/status`) reports `"trigger_health": "degraded"` when Celery is unreachable. The manual trigger still works (runs synchronously in a background thread). The dashboard card shows a warning: "Auto-optimization unavailable — background task system offline."

### Pause on Activity

Any incoming user request sets a pause flag. Current experiment completes cleanly, next one doesn't start. Loop resumes when idle threshold is met again.

**Exemptions from activity tracking:** Requests to `/api/autoresearch/*` endpoints (including "Optimize Now") and `/api/health/*` are excluded from the activity tracker. This prevents the manual trigger from immediately pausing the loop it just started.

## Frontend — Dashboard Card

### RAGAutoresearchCard.jsx

**When idle/stopped:**
- Current best score and config summary
- "Optimize Now" button
- Last experiment timestamp, total experiments / improvements

**When running:**
- Live experiment counter ("Experiment 7 — Phase 1")
- Current hypothesis being tested
- Mini score-over-time chart (like Karpathy's progress.png)
- Last 5 results with keep/discard/crash and deltas
- Pause button

**Family view (interconnector active):**
- Which Nephews are running experiments
- Recent family broadcasts
- Combined experiment count

**Settings (accessible from card):**
- Idle threshold slider
- Auto-start toggle
- Phase limit selector
- Parameter lock toggles

## Family Architecture

### Interconnector Integration

When a Nephew finds a winning config:
1. `ExperimentRun` logged locally with status "keep"
2. Config + research insight broadcast as `InterconnectorLearning` record
3. Other Nephews receive broadcast, queue it as a "family suggestion"
4. Family suggestions enter the normal experiment loop (eval against local data)
5. If it improves local score too -> keep, `ResearchConfig.source = "family_broadcast"`

### Uncle Claude Role (optional, tiered)

**Periodic review (12h Celery task, existing slot):**
- Reads combined experiment log from all connected Nephews
- Analyzes patterns: what's working, what's not, where are Nephews stuck
- Appends insights to `research_program.md` under `## Uncle Claude Directives`
- Can suggest specific experiments or parameter regions to explore

**Guardian review (Phase 2/3 changes):**
- Phase 1 wins: auto-promoted (low risk, query-time only)
- Phase 2 wins: submitted to a new `claude_advisor_service.review_config_change(param, old_value, new_value, score_delta, experiment_history)` method. The existing `review_change()` is designed for code diffs; we need a config-aware variant with a prompt tuned for RAG parameter evaluation.
- Phase 3 wins: require explicit "proceed" from Uncle (never auto-promoted)

**Offline fallback:**
- When Uncle Claude is unavailable (no API key, budget exhausted, HIPAA mode):
- All phases auto-promote based on eval score alone
- Phase 3 still requires user opt-in from settings
- Research program uses only local insights + family broadcasts

## Experiment Agent Specification

The `rag_experiment_agent.py` is the most complex new component — it uses the local LLM to intelligently select experiments rather than random search.

### Agent Prompt Structure

```
System: You are a RAG optimization researcher. You will be given:
1. The current RAG configuration (JSON)
2. The last 20 experiment results (parameter, old→new, score delta, status)
3. The research program directives
4. The current phase and available parameters

Your task: propose exactly ONE parameter change that you believe will
improve the composite quality score. Return JSON:
{
  "parameter": "top_k",
  "new_value": 8,
  "hypothesis": "Increasing top_k from 5 to 8 should improve completeness
                 by providing more candidate chunks for context selection"
}

Rules:
- Only propose parameters valid for the current phase
- Do not repeat an exact experiment that was already tried (check history)
- Explain your reasoning in the hypothesis
- If you see a pattern in the history, exploit it
- If nothing has worked recently, try a different direction
```

### History Formatting

The last 20 experiments are formatted as a compact table for the LLM:
```
#  | param              | 5→8    | delta  | status
1  | top_k              | 5→8    | +0.12  | keep
2  | top_k              | 8→12   | -0.03  | discard
3  | dedup_threshold     | 0.85→0.75 | +0.05 | keep
...
```

### Output Parsing

Agent output is parsed as JSON. If parsing fails (malformed output), the system falls back to a simple heuristic: pick a random untried parameter value within the current phase. This ensures the loop never stalls on an LLM output error.

### Model

Uses the active Ollama model (same one serving user queries). Runs with `temperature=0.7` for creative exploration (vs `temperature=0.0` for the judge). If a separate `rag_autoresearch_agent_model` setting is configured, uses that instead.

## Atomic Config Promotion

When an experiment wins ("keep"), the config swap must be atomic to prevent user queries from hitting a partially-applied state:

1. New `ResearchConfig` record written to DB with `is_active=False`
2. Within a single DB transaction: set old active config `is_active=False`, set new config `is_active=True`
3. Update in-memory config cache atomically (Python dict swap, not mutation)
4. The thread-local experiment context is cleared — subsequent user queries read the new active config from the cache

User queries in progress during the swap continue with their already-loaded parameters (thread-local snapshot). Only new queries pick up the new config.

## Safety & Guardrails

### Config Rollback
- Every `ResearchConfig` snapshot preserved in DB
- "Reset to defaults" reverts to factory settings
- "Revert to previous" rolls back last promotion

### Crash Protection
- 3 consecutive crashes -> revert to last good config, pause loop
- Agent logs crash reason, avoids that parameter region
- Max experiment duration: 5 minutes

### Resource Limits
- Max LLM calls per experiment: 200 (configurable)
- Memory guard: pause if system RAM > 90%
- Disk guard: skip Phase 2 if shadow re-index > 500MB

### Protected Parameters
- `PROTECTED_RAG_PARAMS` list — params autoresearch cannot touch
- Users can lock any parameter from settings UI

### Interconnector Safety
- Family configs are suggestions, not directives — always evaluated locally
- Node ID validation prevents spoofed broadcasts
- API key required for directive receipt
- **Family broadcast deduplication:** Before queuing a family suggestion for eval, check if an `ExperimentRun` with the same `parameter_changed` + `new_value` already exists locally. If so, skip (already tried). This prevents redundant evaluation when multiple Nephews independently discover the same winning config.

## File Manifest

### New Files

```
backend/services/rag_autoresearch_service.py  -- Orchestrator (experiment loop, idle detection, pause/resume)
backend/services/rag_eval_harness.py          -- Eval pair generation + LLM-as-judge scoring
backend/services/rag_experiment_agent.py      -- LLM-driven hypothesis engine (reads history, proposes experiments)
backend/api/rag_autoresearch_api.py           -- REST endpoints + Socket.IO event emitters
backend/tasks/rag_autoresearch_tasks.py       -- Celery tasks (idle checker, scheduled runs, event-triggered)
backend/utils/experiment_context.py           -- Thread-local experiment config injection
frontend/src/components/dashboard/RAGAutoresearchCard.jsx  -- Dashboard card (status, history, chart, controls)
frontend/src/api/ragAutoresearchService.js    -- API client for autoresearch endpoints
data/research_program.md                      -- Agent instructions (the "program.md")
data/autoresearch/shadow_index/               -- Shadow corpus index directory (auto-created)
```

### Modified Files

```
backend/app.py                                -- Register blueprint, add activity tracker middleware
backend/models.py                             -- Add ExperimentRun, EvalPair, ResearchConfig tables; extend SelfImprovementRun trigger values
backend/config.py                             -- Add autoresearch config constants + PROTECTED_RAG_PARAMS
backend/celery_app.py                         -- Register autoresearch Celery Beat tasks
backend/services/indexing_service.py          -- Add post-index hook to mark eval stale; add experiment config injection point in search_with_llamaindex()
backend/services/claude_advisor_service.py    -- Add review_config_change() method for Phase 2/3 guardian review
backend/tasks/self_improvement_tasks.py       -- Wire uncle review of experiment logs into existing 12h task
backend/utils/unified_index_manager.py        -- Add shadow index support (separate StorageContext + lock)
frontend/src/pages/DashboardPage.jsx          -- Add RAGAutoresearchCard
frontend/src/pages/SettingsPage.jsx           -- Add autoresearch settings section (A.I. tab)
```

### Existing Infrastructure Leveraged (read-only, no modifications)

```
backend/services/unified_chat_engine.py       -- Eval queries flow through real retrieval path (via experiment_context)
backend/utils/hybrid_rag_pipeline.py          -- Used when query_expansion/reranking/hybrid_search enabled
backend/api/interconnector_api.py             -- Family broadcast of winning configs
backend/utils/enhanced_rag_chunking.py        -- ChunkingStrategy dataclass for Phase 2 params
backend/api/rag_debug_api.py                  -- Existing metrics complement eval harness
backend/socketio_instance.py                  -- Real-time experiment update events
```
