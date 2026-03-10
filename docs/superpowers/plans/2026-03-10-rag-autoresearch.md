# RAG Autoresearch Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous RAG optimization loop that runs experiments while the system is idle, measuring and improving retrieval quality without human intervention.

**Architecture:** Karpathy autoresearch pattern — an LLM agent proposes parameter changes, an eval harness scores them via LLM-as-judge, and a keep/revert loop advances the config. Thread-local experiment context injects configs into the real retrieval path. Celery Beat handles idle detection and scheduling. Interconnector broadcasts winning configs to family instances.

**Tech Stack:** Flask, SQLAlchemy/Alembic, Celery Beat, LlamaIndex, Ollama, React/MUI, Socket.IO

**Spec:** `docs/superpowers/specs/2026-03-10-rag-autoresearch-design.md`

---

## Chunk 1: Database Models & Migration

### Task 1: Add ExperimentRun, EvalPair, ResearchConfig models

**Files:**
- Modify: `backend/models.py` (after line ~1712, the last model class `WordPressPage`)
- Create: `backend/migrations/versions/<auto>_add_autoresearch_tables.py` (via flask db migrate)
- Test: `backend/tests/test_autoresearch_models.py`

- [ ] **Step 1: Write model tests**

Create `backend/tests/test_autoresearch_models.py`:

```python
"""Tests for RAG Autoresearch database models."""
import uuid
import pytest
from datetime import datetime
from backend.models import db, ExperimentRun, EvalPair, ResearchConfig


class TestExperimentRun:
    def test_create_experiment_run(self, app):
        """ExperimentRun can be created with required fields."""
        with app.app_context():
            run = ExperimentRun(
                id=str(uuid.uuid4()),
                run_tag="mar10-test",
                phase=1,
                parameter_changed="top_k",
                old_value="5",
                new_value="8",
                hypothesis="Increasing top_k should improve completeness",
                composite_score=3.5,
                baseline_score=3.2,
                delta=0.3,
                status="keep",
                duration_seconds=45.2,
            )
            db.session.add(run)
            db.session.commit()
            fetched = db.session.get(ExperimentRun, run.id)
            assert fetched.parameter_changed == "top_k"
            assert fetched.delta == 0.3
            assert fetched.status == "keep"
            assert fetched.node_id is None  # nullable for standalone

    def test_experiment_run_with_eval_details(self, app):
        """ExperimentRun stores JSON eval_details."""
        with app.app_context():
            run = ExperimentRun(
                id=str(uuid.uuid4()),
                phase=1,
                parameter_changed="top_k",
                old_value="5",
                new_value="8",
                composite_score=3.5,
                baseline_score=3.2,
                delta=0.3,
                status="keep",
                eval_details={"q1": {"relevance": 4, "grounding": 3, "completeness": 4}},
            )
            db.session.add(run)
            db.session.commit()
            fetched = db.session.get(ExperimentRun, run.id)
            assert fetched.eval_details["q1"]["relevance"] == 4


class TestEvalPair:
    def test_create_eval_pair(self, app):
        """EvalPair can be created with required fields."""
        with app.app_context():
            pair = EvalPair(
                id=str(uuid.uuid4()),
                eval_generation_id="gen-001",
                question="How does the chat streaming pipeline work?",
                expected_answer="It uses Socket.IO via unified_chat_engine.py",
                source_chunk_hash="a" * 64,
                corpus_type="code",
            )
            db.session.add(pair)
            db.session.commit()
            fetched = db.session.get(EvalPair, pair.id)
            assert fetched.corpus_type == "code"
            assert fetched.quality_score is None  # nullable


class TestResearchConfig:
    def test_create_research_config(self, app):
        """ResearchConfig stores JSON params and tracks active state."""
        with app.app_context():
            config = ResearchConfig(
                id=str(uuid.uuid4()),
                params={"top_k": 5, "similarity_threshold": 0.85},
                composite_score=3.2,
                is_active=True,
                source="local",
            )
            db.session.add(config)
            db.session.commit()
            fetched = db.session.get(ResearchConfig, config.id)
            assert fetched.params["top_k"] == 5
            assert fetched.is_active is True
            assert fetched.source == "local"

    def test_only_one_active_config(self, app):
        """Only one ResearchConfig should be active at a time (enforced by app logic)."""
        with app.app_context():
            c1 = ResearchConfig(
                id=str(uuid.uuid4()),
                params={"top_k": 5},
                composite_score=3.0,
                is_active=True,
                source="local",
            )
            c2 = ResearchConfig(
                id=str(uuid.uuid4()),
                params={"top_k": 8},
                composite_score=3.5,
                is_active=False,
                source="local",
            )
            db.session.add_all([c1, c2])
            db.session.commit()
            active = ResearchConfig.query.filter_by(is_active=True).all()
            assert len(active) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/test_autoresearch_models.py -v
```
Expected: ImportError — `ExperimentRun`, `EvalPair`, `ResearchConfig` not found in models.

- [ ] **Step 3: Add models to backend/models.py**

Add after the `WordPressPage` class (around line 1712):

```python
# ---------------------------------------------------------------------------
# RAG Autoresearch Models
# ---------------------------------------------------------------------------

class ExperimentRun(db.Model):
    """Tracks individual autoresearch experiment results."""
    __tablename__ = "experiment_runs"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_tag = db.Column(db.String(100), nullable=True, index=True)
    phase = db.Column(db.Integer, nullable=False, default=1)
    parameter_changed = db.Column(db.String(200), nullable=False)
    old_value = db.Column(db.String(500), nullable=True)
    new_value = db.Column(db.String(500), nullable=False)
    hypothesis = db.Column(db.Text, nullable=True)
    composite_score = db.Column(db.Float, nullable=False)
    baseline_score = db.Column(db.Float, nullable=True)
    delta = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), nullable=False, default="discard")  # keep, discard, crash
    eval_details = db.Column(db.JSON, nullable=True)
    duration_seconds = db.Column(db.Float, nullable=True)
    node_id = db.Column(db.String(36), nullable=True)  # nullable for standalone instances
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id, "run_tag": self.run_tag, "phase": self.phase,
            "parameter_changed": self.parameter_changed,
            "old_value": self.old_value, "new_value": self.new_value,
            "hypothesis": self.hypothesis,
            "composite_score": self.composite_score,
            "baseline_score": self.baseline_score, "delta": self.delta,
            "status": self.status, "eval_details": self.eval_details,
            "duration_seconds": self.duration_seconds,
            "node_id": self.node_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class EvalPair(db.Model):
    """Auto-generated Q&A pairs for RAG evaluation."""
    __tablename__ = "eval_pairs"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    eval_generation_id = db.Column(db.String(50), nullable=True, index=True)
    question = db.Column(db.Text, nullable=False)
    expected_answer = db.Column(db.Text, nullable=False)
    source_doc_id = db.Column(db.Integer, db.ForeignKey("documents.id"), nullable=True)
    source_chunk_hash = db.Column(db.String(64), nullable=True)
    corpus_type = db.Column(db.String(20), nullable=True)  # code, knowledge, client
    quality_score = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    source_document = db.relationship("Document", backref="eval_pairs", lazy=True)

    def to_dict(self):
        return {
            "id": self.id, "eval_generation_id": self.eval_generation_id,
            "question": self.question, "expected_answer": self.expected_answer,
            "source_doc_id": self.source_doc_id,
            "source_chunk_hash": self.source_chunk_hash,
            "corpus_type": self.corpus_type,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ResearchConfig(db.Model):
    """Snapshot of RAG configuration with its eval score."""
    __tablename__ = "research_configs"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    params = db.Column(db.JSON, nullable=False)
    composite_score = db.Column(db.Float, nullable=True)
    is_active = db.Column(db.Boolean, default=False, index=True)
    promoted_at = db.Column(db.DateTime, nullable=True)
    source = db.Column(db.String(30), nullable=True)  # local, family_broadcast, uncle_directive
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id, "params": self.params,
            "composite_score": self.composite_score,
            "is_active": self.is_active,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
```

Also add `import uuid` at top of file if not already present.

- [ ] **Step 4: Generate and apply migration**

```bash
cd /home/llamax1/LLAMAX7/backend && source venv/bin/activate && flask db migrate -m "add autoresearch tables" && flask db upgrade
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/test_autoresearch_models.py -v
```
Expected: All 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/models.py backend/migrations/versions/*autoresearch* backend/tests/test_autoresearch_models.py
git commit -m "feat(autoresearch): add ExperimentRun, EvalPair, ResearchConfig models"
```

---

## Chunk 2: Config & Experiment Context Plumbing

### Task 2: Add autoresearch config constants

**Files:**
- Modify: `backend/config.py` (after line ~109, near CHUNK_SIMILARITY_THRESHOLD)

- [ ] **Step 1: Add config constants to backend/config.py**

Add after the existing RAG config section (around line 109):

```python
# RAG Autoresearch configuration
AUTORESEARCH_ENABLED = os.environ.get("GUAARDVARK_AUTORESEARCH_ENABLED", "true").lower() == "true"
AUTORESEARCH_IDLE_MINUTES = int(os.environ.get("GUAARDVARK_AUTORESEARCH_IDLE_MINUTES", "10"))
AUTORESEARCH_MAX_EXPERIMENT_DURATION = 300  # 5 minutes, matching Karpathy's time budget
AUTORESEARCH_MAX_LLM_CALLS_PER_EXPERIMENT = 200
AUTORESEARCH_PHASE_PLATEAU_THRESHOLD = 10  # consecutive discards before phase advance
AUTORESEARCH_MIN_CORPUS_SIZE = 10  # minimum indexed documents to enable
AUTORESEARCH_SHADOW_CORPUS_SIZE = 100  # documents in shadow eval corpus
AUTORESEARCH_EVAL_PAIR_TARGET = 100  # target eval pairs per generation
AUTORESEARCH_STALENESS_SAMPLE_RATE = 0.1  # fraction of pairs to spot-check
AUTORESEARCH_STALENESS_THRESHOLD = 0.2  # fraction of stale pairs triggering regen

# Default RAG experiment parameters
AUTORESEARCH_DEFAULT_PARAMS = {
    # Phase 1 — query-time
    "top_k": 5,
    "dedup_threshold": 0.85,
    "context_window_chunks": 3,
    "reranking_enabled": False,
    "query_expansion": False,
    "hybrid_search_alpha": 0.0,
    # Phase 2 — index-time
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "use_semantic_splitting": False,
    "use_hierarchical_splitting": False,
    "extract_entities": False,
    "preserve_structure": False,
}

PROTECTED_RAG_PARAMS = []  # params autoresearch cannot touch (user-configurable)
```

- [ ] **Step 2: Commit**

```bash
git add backend/config.py
git commit -m "feat(autoresearch): add autoresearch config constants"
```

### Task 3: Create thread-local experiment context

**Files:**
- Create: `backend/utils/experiment_context.py`
- Test: `backend/tests/test_experiment_context.py`

- [ ] **Step 1: Write tests for experiment context**

Create `backend/tests/test_experiment_context.py`:

```python
"""Tests for thread-local experiment config injection."""
import threading
from backend.utils.experiment_context import (
    set_experiment_config,
    get_experiment_config,
    clear_experiment_config,
)


def test_get_returns_none_by_default():
    clear_experiment_config()
    assert get_experiment_config() is None


def test_set_and_get_config():
    config = {"top_k": 10, "dedup_threshold": 0.75}
    set_experiment_config(config)
    assert get_experiment_config() == config
    clear_experiment_config()


def test_clear_removes_config():
    set_experiment_config({"top_k": 10})
    clear_experiment_config()
    assert get_experiment_config() is None


def test_thread_isolation():
    """Config set in one thread is not visible in another."""
    set_experiment_config({"top_k": 10})
    result = {}

    def check_other_thread():
        result["config"] = get_experiment_config()

    t = threading.Thread(target=check_other_thread)
    t.start()
    t.join()
    assert result["config"] is None
    clear_experiment_config()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/test_experiment_context.py -v
```
Expected: ImportError — module not found.

- [ ] **Step 3: Create backend/utils/experiment_context.py**

```python
"""Thread-local experiment config for RAG autoresearch.

During eval runs, the autoresearch orchestrator sets experiment parameters
via set_experiment_config(). The retrieval path (search_with_llamaindex,
_retrieve_rag_context) checks get_experiment_config() and applies overrides
if present. Outside of eval runs, get_experiment_config() returns None and
the default config is used — zero impact on normal user queries.
"""
import threading

_experiment_config = threading.local()


def set_experiment_config(config: dict):
    """Set experiment params for current thread."""
    _experiment_config.params = config


def get_experiment_config():
    """Get experiment params, or None if not in an experiment."""
    return getattr(_experiment_config, "params", None)


def clear_experiment_config():
    """Remove experiment config from current thread."""
    _experiment_config.params = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/test_experiment_context.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/utils/experiment_context.py backend/tests/test_experiment_context.py
git commit -m "feat(autoresearch): add thread-local experiment context injection"
```

### Task 4: Wire experiment context into retrieval path

**Files:**
- Modify: `backend/services/indexing_service.py` (function `search_with_llamaindex` at line ~436)
- Test: `backend/tests/test_experiment_context_integration.py`

- [ ] **Step 1: Write integration test**

Create `backend/tests/test_experiment_context_integration.py`:

```python
"""Test that experiment context overrides retrieval parameters."""
from unittest.mock import patch, MagicMock
from backend.utils.experiment_context import (
    set_experiment_config,
    clear_experiment_config,
)


def test_search_uses_experiment_top_k(app):
    """When experiment context is set, search_with_llamaindex uses experiment top_k."""
    with app.app_context():
        set_experiment_config({"top_k": 12, "context_window_chunks": 5})
        try:
            from backend.services.indexing_service import search_with_llamaindex
            # We mock the actual retriever to avoid needing a real index
            with patch("backend.services.indexing_service.get_or_create_index") as mock_idx:
                mock_retriever = MagicMock()
                mock_retriever.retrieve.return_value = []
                mock_idx.return_value = MagicMock()
                mock_idx.return_value.as_retriever.return_value = mock_retriever
                search_with_llamaindex("test query", max_chunks=3)
                # Verify the retriever was called with experiment top_k
                call_kwargs = mock_idx.return_value.as_retriever.call_args
                if call_kwargs:
                    assert call_kwargs[1].get("similarity_top_k", 5) == 12
        finally:
            clear_experiment_config()


def test_search_uses_defaults_without_experiment(app):
    """Without experiment context, default parameters are used."""
    clear_experiment_config()
    with app.app_context():
        from backend.services.indexing_service import search_with_llamaindex
        with patch("backend.services.indexing_service.get_or_create_index") as mock_idx:
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = []
            mock_idx.return_value = MagicMock()
            mock_idx.return_value.as_retriever.return_value = mock_retriever
            search_with_llamaindex("test query", max_chunks=3)
            # Default top_k should be used (5 or whatever the function default is)
            call_kwargs = mock_idx.return_value.as_retriever.call_args
            if call_kwargs:
                top_k = call_kwargs[1].get("similarity_top_k", 5)
                assert top_k <= 10  # not the experiment value of 12
```

- [ ] **Step 2: Modify search_with_llamaindex in indexing_service.py**

At the top of `search_with_llamaindex()` (line ~436), add the experiment config check:

```python
from backend.utils.experiment_context import get_experiment_config

def search_with_llamaindex(query, max_chunks=5, project_id=None):
    # Check for active experiment config overrides
    exp_config = get_experiment_config()
    if exp_config:
        max_chunks = exp_config.get("context_window_chunks", max_chunks)
        effective_top_k = exp_config.get("top_k", max_chunks)
    else:
        effective_top_k = max_chunks

    # ... rest of existing function, using effective_top_k for retriever
    # and max_chunks for final slicing
```

The exact modification depends on how the existing function structures its retriever call. Read the function body, find where `similarity_top_k` or the retriever is configured, and inject `effective_top_k` there. Keep `max_chunks` for the final result slicing (`results[:max_chunks]`).

- [ ] **Step 3: Run tests**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/test_experiment_context_integration.py -v
```

- [ ] **Step 4: Commit**

```bash
git add backend/services/indexing_service.py backend/tests/test_experiment_context_integration.py
git commit -m "feat(autoresearch): wire experiment context into retrieval path"
```

---

## Chunk 3: Eval Harness

### Task 5: Create the eval harness service

**Files:**
- Create: `backend/services/rag_eval_harness.py`
- Test: `backend/tests/test_rag_eval_harness.py`

- [ ] **Step 1: Write eval harness tests**

Create `backend/tests/test_rag_eval_harness.py`:

```python
"""Tests for RAG eval harness — eval pair generation and LLM-as-judge scoring."""
import pytest
from unittest.mock import patch, MagicMock
from backend.services.rag_eval_harness import RAGEvalHarness


class TestEvalPairGeneration:
    def test_generate_eval_pair_from_chunk(self):
        """Given a text chunk, generates a question and expected answer."""
        harness = RAGEvalHarness()
        chunk_text = "The unified_chat_engine uses Socket.IO for real-time streaming."
        mock_response = '{"question": "How does the chat engine handle streaming?", "expected_answer": "It uses Socket.IO for real-time streaming via unified_chat_engine."}'

        with patch.object(harness, "_call_llm", return_value=mock_response):
            result = harness.generate_eval_pair(chunk_text, "code")
            assert "question" in result
            assert "expected_answer" in result
            assert result["corpus_type"] == "code"

    def test_generate_eval_pair_handles_malformed_llm_output(self):
        """Gracefully handles unparseable LLM output."""
        harness = RAGEvalHarness()
        with patch.object(harness, "_call_llm", return_value="not json"):
            result = harness.generate_eval_pair("some text", "code")
            assert result is None

    def test_minimum_corpus_check(self, app):
        """Returns False if corpus is below minimum threshold."""
        with app.app_context():
            harness = RAGEvalHarness()
            assert harness.has_sufficient_corpus() is False  # empty DB


class TestLLMJudge:
    def test_score_response_returns_composite(self):
        """LLM-as-judge returns relevance, grounding, completeness, composite."""
        harness = RAGEvalHarness()
        mock_judgment = '{"relevance": 4, "grounding": 5, "completeness": 3}'

        with patch.object(harness, "_call_llm", return_value=mock_judgment):
            score = harness.score_response(
                question="How does streaming work?",
                expected_answer="Socket.IO streaming",
                actual_response="The system uses Socket.IO for streaming.",
                retrieved_chunks=["chunk about Socket.IO"],
            )
            assert score["relevance"] == 4
            assert score["grounding"] == 5
            assert score["completeness"] == 3
            assert 1.0 <= score["composite"] <= 5.0

    def test_score_response_handles_malformed_judgment(self):
        """Returns default low scores on parse failure."""
        harness = RAGEvalHarness()
        with patch.object(harness, "_call_llm", return_value="garbage"):
            score = harness.score_response("q", "a", "r", [])
            assert score["composite"] == 1.0  # worst score

    def test_run_full_eval(self, app):
        """Full eval runs all eval pairs and returns average composite score."""
        with app.app_context():
            harness = RAGEvalHarness()
            # Mock eval pairs in DB and scoring
            with patch.object(harness, "_get_active_eval_pairs") as mock_pairs, \
                 patch.object(harness, "_eval_single_pair") as mock_eval:
                mock_pairs.return_value = [
                    {"id": "1", "question": "q1", "expected_answer": "a1"},
                    {"id": "2", "question": "q2", "expected_answer": "a2"},
                ]
                mock_eval.side_effect = [
                    {"composite": 4.0, "relevance": 4, "grounding": 4, "completeness": 4},
                    {"composite": 3.0, "relevance": 3, "grounding": 3, "completeness": 3},
                ]
                result = harness.run_full_eval(config={"top_k": 5})
                assert result["composite_score"] == 3.5
                assert result["num_pairs"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/test_rag_eval_harness.py -v
```

- [ ] **Step 3: Implement rag_eval_harness.py**

Create `backend/services/rag_eval_harness.py`:

```python
"""RAG Eval Harness — the 'prepare.py' of Guaardvark autoresearch.

Generates eval Q&A pairs from indexed documents and scores RAG responses
using LLM-as-judge. The composite quality score (1.0-5.0, higher=better)
is the single metric for the autoresearch keep/revert loop.
"""
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional

from backend.config import (
    AUTORESEARCH_EVAL_PAIR_TARGET,
    AUTORESEARCH_MIN_CORPUS_SIZE,
    AUTORESEARCH_STALENESS_SAMPLE_RATE,
    AUTORESEARCH_STALENESS_THRESHOLD,
)

logger = logging.getLogger(__name__)

# --- Prompts ---

EVAL_PAIR_GENERATION_PROMPT = """You are generating evaluation questions for a RAG (Retrieval-Augmented Generation) system.

Given the following text chunk, generate ONE factual question that a user would ask, and the correct answer based ONLY on this text.

Text chunk:
{chunk_text}

Return ONLY valid JSON:
{{"question": "your question here", "expected_answer": "the answer from the text"}}"""

JUDGE_PROMPT = """You are evaluating the quality of a RAG system's response.

Question: {question}
Expected Answer: {expected_answer}
Actual Response: {actual_response}
Retrieved Context Chunks:
{chunks_text}

Score each dimension from 1-5 (5=best):
- relevance: Are the retrieved chunks relevant to the question?
- grounding: Is the response supported by the retrieved chunks (not hallucinated)?
- completeness: Does the response fully address the question?

Return ONLY valid JSON:
{{"relevance": N, "grounding": N, "completeness": N}}"""


class RAGEvalHarness:
    """Immutable eval harness for autoresearch experiments."""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        """Get the Ollama LLM instance (lazy-loaded)."""
        if self._llm is None:
            try:
                from flask import current_app
                self._llm = current_app.config.get("LLAMA_INDEX_LLM")
            except RuntimeError:
                pass
            if self._llm is None:
                from backend.services.llm_service import get_llm
                self._llm = get_llm()
        return self._llm

    def _call_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """Call the local LLM with the given prompt."""
        llm = self._get_llm()
        if llm is None:
            return ""
        try:
            response = llm.complete(prompt, temperature=temperature)
            return str(response).strip()
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ""

    def has_sufficient_corpus(self) -> bool:
        """Check if enough documents are indexed for meaningful eval."""
        from backend.models import Document, db
        count = db.session.query(Document).count()
        return count >= AUTORESEARCH_MIN_CORPUS_SIZE

    def generate_eval_pair(self, chunk_text: str, corpus_type: str) -> Optional[dict]:
        """Generate a Q&A eval pair from a text chunk."""
        prompt = EVAL_PAIR_GENERATION_PROMPT.format(chunk_text=chunk_text[:2000])
        response = self._call_llm(prompt, temperature=0.3)
        try:
            parsed = json.loads(response)
            if "question" in parsed and "expected_answer" in parsed:
                parsed["corpus_type"] = corpus_type
                parsed["source_chunk_hash"] = hashlib.sha256(
                    chunk_text.encode()
                ).hexdigest()
                return parsed
        except (json.JSONDecodeError, KeyError):
            pass
        return None

    def generate_eval_set(self, target_count: int = None):
        """Generate a full eval set from indexed documents.

        Returns list of eval pair dicts ready for DB insertion.
        """
        if target_count is None:
            target_count = AUTORESEARCH_EVAL_PAIR_TARGET

        from backend.models import Document, db
        import random

        documents = Document.query.all()
        if len(documents) < AUTORESEARCH_MIN_CORPUS_SIZE:
            logger.warning(
                f"Insufficient corpus: {len(documents)} docs < {AUTORESEARCH_MIN_CORPUS_SIZE} minimum"
            )
            return []

        # Sample documents, stratified by any available type info
        sampled = random.sample(documents, min(len(documents), target_count * 2))
        generation_id = f"gen-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        pairs = []
        for doc in sampled:
            if len(pairs) >= target_count:
                break
            # Use document content or title as the chunk text
            chunk_text = getattr(doc, "content", None) or getattr(doc, "title", "") or ""
            if len(chunk_text) < 50:
                continue
            corpus_type = self._detect_corpus_type(doc)
            pair = self.generate_eval_pair(chunk_text, corpus_type)
            if pair:
                pair["eval_generation_id"] = generation_id
                pair["source_doc_id"] = doc.id
                pairs.append(pair)

        logger.info(f"Generated {len(pairs)} eval pairs (generation: {generation_id})")
        return pairs

    def _detect_corpus_type(self, document) -> str:
        """Detect corpus type from document metadata."""
        name = getattr(document, "title", "") or getattr(document, "name", "") or ""
        name_lower = name.lower()
        if any(ext in name_lower for ext in [".py", ".js", ".jsx", ".ts", ".tsx", ".sh", ".sql"]):
            return "code"
        if any(kw in name_lower for kw in ["client", "project", "brief", "proposal"]):
            return "client"
        return "knowledge"

    def score_response(
        self,
        question: str,
        expected_answer: str,
        actual_response: str,
        retrieved_chunks: list,
    ) -> dict:
        """LLM-as-judge scoring. Returns {relevance, grounding, completeness, composite}."""
        chunks_text = "\n---\n".join(
            str(c)[:500] for c in (retrieved_chunks or [])
        )
        prompt = JUDGE_PROMPT.format(
            question=question,
            expected_answer=expected_answer,
            actual_response=actual_response,
            chunks_text=chunks_text or "(no chunks retrieved)",
        )
        response = self._call_llm(prompt, temperature=0.0)
        try:
            parsed = json.loads(response)
            relevance = max(1, min(5, int(parsed.get("relevance", 1))))
            grounding = max(1, min(5, int(parsed.get("grounding", 1))))
            completeness = max(1, min(5, int(parsed.get("completeness", 1))))
            composite = (relevance + grounding + completeness) / 3.0
            return {
                "relevance": relevance,
                "grounding": grounding,
                "completeness": completeness,
                "composite": round(composite, 3),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            return {"relevance": 1, "grounding": 1, "completeness": 1, "composite": 1.0}

    def _get_active_eval_pairs(self) -> list:
        """Load active eval pairs from DB."""
        from backend.models import EvalPair
        pairs = EvalPair.query.order_by(EvalPair.created_at.desc()).all()
        return [p.to_dict() for p in pairs]

    def _eval_single_pair(self, pair: dict, config: dict) -> dict:
        """Run a single eval pair through the RAG pipeline and score it."""
        from backend.utils.experiment_context import (
            set_experiment_config,
            clear_experiment_config,
        )
        from backend.services.indexing_service import search_with_llamaindex

        try:
            set_experiment_config(config)
            # Query the real retrieval path
            results = search_with_llamaindex(
                pair["question"],
                max_chunks=config.get("context_window_chunks", 3),
            )
            retrieved_chunks = [r.get("text", "") for r in results] if results else []

            # Generate response using LLM with retrieved context
            context = "\n".join(retrieved_chunks)
            response_prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {pair['question']}\n\nAnswer:"
            actual_response = self._call_llm(response_prompt, temperature=0.0)

            return self.score_response(
                question=pair["question"],
                expected_answer=pair["expected_answer"],
                actual_response=actual_response,
                retrieved_chunks=retrieved_chunks,
            )
        finally:
            clear_experiment_config()

    def run_full_eval(self, config: dict) -> dict:
        """Run all eval pairs through the RAG pipeline with given config.

        Returns {composite_score, num_pairs, details: [...]}
        """
        pairs = self._get_active_eval_pairs()
        if not pairs:
            return {"composite_score": 0.0, "num_pairs": 0, "details": []}

        details = []
        total_composite = 0.0
        for pair in pairs:
            score = self._eval_single_pair(pair, config)
            score["eval_pair_id"] = pair["id"]
            details.append(score)
            total_composite += score["composite"]

        avg_composite = total_composite / len(pairs) if pairs else 0.0
        return {
            "composite_score": round(avg_composite, 4),
            "num_pairs": len(pairs),
            "details": details,
        }

    def is_stale(self) -> bool:
        """Check if eval pairs need regeneration by sampling chunk hashes."""
        import random
        from backend.models import EvalPair

        pairs = EvalPair.query.filter(
            EvalPair.source_chunk_hash.isnot(None)
        ).all()
        if not pairs:
            return True

        sample_size = max(1, int(len(pairs) * AUTORESEARCH_STALENESS_SAMPLE_RATE))
        sample = random.sample(pairs, min(sample_size, len(pairs)))

        stale_count = 0
        for pair in sample:
            # Check if source doc still exists and chunk hash matches
            if pair.source_document is None:
                stale_count += 1
                continue
            # In a full implementation, re-hash the chunk from the index
            # and compare. For now, check document existence.

        stale_ratio = stale_count / len(sample) if sample else 1.0
        return stale_ratio > AUTORESEARCH_STALENESS_THRESHOLD
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/test_rag_eval_harness.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/services/rag_eval_harness.py backend/tests/test_rag_eval_harness.py
git commit -m "feat(autoresearch): implement RAG eval harness with LLM-as-judge"
```

---

## Chunk 4: Experiment Agent & Research Program

### Task 6: Create the research program file

**Files:**
- Create: `data/research_program.md`

- [ ] **Step 1: Create data/research_program.md**

```markdown
# RAG Autoresearch Program

## Your Role
You are an autonomous RAG optimization researcher for Guaardvark. You read
experiment history, form hypotheses, propose ONE parameter change per cycle,
and evaluate results. You work indefinitely without human intervention.

## Rules
- Modify only parameters listed in the current phase
- ONE change per experiment (isolate variables) unless combining near-misses
- If 3 consecutive experiments crash, revert to last known good config
- Prefer simplicity: if two configs score equally, keep the simpler one
- Log your reasoning in the hypothesis field

## Phase 1 Parameters (query-time, no re-indexing)
- top_k (1-20): chunks retrieved from vector store
- dedup_threshold (0.5-0.98): post-retrieval deduplication cutoff
- context_window_chunks (1-10): chunks included in LLM context
- reranking_enabled (bool): re-rank by relevance
- query_expansion (bool): expand query with synonyms
- hybrid_search_alpha (0.0-1.0): vector vs keyword blend

## Phase 2 Parameters (index-time, uses shadow corpus)
- chunk_size (200-3000): tokens per chunk
- chunk_overlap (0-500): overlap between chunks
- use_semantic_splitting (bool): semantic boundary splitting
- use_hierarchical_splitting (bool): parent-child chunks
- extract_entities (bool): entity extraction
- preserve_structure (bool): maintain document structure

## Strategy
1. Start with Phase 1 parameters — they are free to test
2. Try large changes first to find the ballpark, then fine-tune
3. When you see a pattern (e.g., higher top_k always helps), push further
4. Consider corpus composition — code retrieval may want different params
5. If stuck after many discards, try combining two previous near-misses
6. Check if params interact: top_k and context_window_chunks are related

## What Success Looks Like
Higher composite score = better. A 0.1 improvement is significant.
A 0.01 improvement that adds complexity (enabling a feature) may still
be worth it if the feature is simple. Track trends, not just single results.
```

- [ ] **Step 2: Commit**

```bash
git add data/research_program.md
git commit -m "feat(autoresearch): add default research program"
```

### Task 7: Create the experiment agent

**Files:**
- Create: `backend/services/rag_experiment_agent.py`
- Test: `backend/tests/test_rag_experiment_agent.py`

- [ ] **Step 1: Write experiment agent tests**

Create `backend/tests/test_rag_experiment_agent.py`:

```python
"""Tests for the RAG experiment agent — hypothesis engine."""
import pytest
from unittest.mock import patch, MagicMock
from backend.services.rag_experiment_agent import RAGExperimentAgent


class TestPropose:
    def test_proposes_valid_phase1_experiment(self):
        """Agent proposes a change within Phase 1 parameters."""
        agent = RAGExperimentAgent()
        mock_response = '{"parameter": "top_k", "new_value": 8, "hypothesis": "More chunks may help"}'

        with patch.object(agent, "_call_llm", return_value=mock_response):
            proposal = agent.propose_experiment(
                history=[],
                current_config={"top_k": 5, "dedup_threshold": 0.85},
                phase=1,
            )
            assert proposal["parameter"] == "top_k"
            assert proposal["new_value"] == 8
            assert "hypothesis" in proposal

    def test_handles_malformed_llm_output(self):
        """Falls back to random parameter on parse failure."""
        agent = RAGExperimentAgent()
        with patch.object(agent, "_call_llm", return_value="not json at all"):
            proposal = agent.propose_experiment(
                history=[], current_config={"top_k": 5}, phase=1,
            )
            assert "parameter" in proposal
            assert "new_value" in proposal
            assert "hypothesis" in proposal

    def test_avoids_recently_tried_experiments(self):
        """Agent does not propose the same experiment twice."""
        agent = RAGExperimentAgent()
        history = [
            {"parameter_changed": "top_k", "new_value": "8", "status": "discard"},
            {"parameter_changed": "top_k", "new_value": "10", "status": "discard"},
        ]
        mock_response = '{"parameter": "dedup_threshold", "new_value": 0.75, "hypothesis": "Lower dedup"}'

        with patch.object(agent, "_call_llm", return_value=mock_response):
            proposal = agent.propose_experiment(
                history=history,
                current_config={"top_k": 5, "dedup_threshold": 0.85},
                phase=1,
            )
            # Should not propose top_k=8 or top_k=10 again
            assert not (proposal["parameter"] == "top_k" and proposal["new_value"] in [8, 10])


class TestPhaseTransition:
    def test_should_advance_phase_after_plateau(self):
        """Advances phase after 10 consecutive discards."""
        agent = RAGExperimentAgent()
        history = [{"status": "discard"} for _ in range(10)]
        assert agent.should_advance_phase(history) is True

    def test_should_not_advance_if_recent_keep(self):
        """Does not advance if there was a recent keep."""
        agent = RAGExperimentAgent()
        history = [{"status": "discard"} for _ in range(9)]
        history.append({"status": "keep"})
        assert agent.should_advance_phase(history) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/test_rag_experiment_agent.py -v
```

- [ ] **Step 3: Implement rag_experiment_agent.py**

Create `backend/services/rag_experiment_agent.py`:

```python
"""RAG Experiment Agent — LLM-driven hypothesis engine.

Reads experiment history and research_program.md, proposes the next
parameter change to try. Falls back to random search if LLM output
is unparseable.
"""
import json
import os
import random
import logging
from typing import Optional

from backend.config import (
    AUTORESEARCH_DEFAULT_PARAMS,
    AUTORESEARCH_PHASE_PLATEAU_THRESHOLD,
    PROTECTED_RAG_PARAMS,
)

logger = logging.getLogger(__name__)

# Phase -> parameter names
PHASE_PARAMS = {
    1: ["top_k", "dedup_threshold", "context_window_chunks",
        "reranking_enabled", "query_expansion", "hybrid_search_alpha"],
    2: ["chunk_size", "chunk_overlap", "use_semantic_splitting",
        "use_hierarchical_splitting", "extract_entities", "preserve_structure"],
    3: ["embedding_model"],
}

# Parameter ranges for random fallback
PARAM_RANGES = {
    "top_k": (1, 20, "int"),
    "dedup_threshold": (0.5, 0.98, "float"),
    "context_window_chunks": (1, 10, "int"),
    "reranking_enabled": (False, True, "bool"),
    "query_expansion": (False, True, "bool"),
    "hybrid_search_alpha": (0.0, 1.0, "float"),
    "chunk_size": (200, 3000, "int"),
    "chunk_overlap": (0, 500, "int"),
    "use_semantic_splitting": (False, True, "bool"),
    "use_hierarchical_splitting": (False, True, "bool"),
    "extract_entities": (False, True, "bool"),
    "preserve_structure": (False, True, "bool"),
}

AGENT_PROMPT = """You are a RAG optimization researcher. Based on the experiment history
and research program, propose ONE parameter change.

Current config: {current_config}
Phase {phase} parameters you can change: {available_params}
Research program directives:
{research_program}

Last {history_count} experiments:
{history_table}

Propose exactly ONE change. Return ONLY valid JSON:
{{"parameter": "param_name", "new_value": <value>, "hypothesis": "your reasoning"}}"""


class RAGExperimentAgent:
    """Proposes experiments using LLM reasoning + random fallback."""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            try:
                from flask import current_app
                self._llm = current_app.config.get("LLAMA_INDEX_LLM")
            except RuntimeError:
                pass
            if self._llm is None:
                try:
                    from backend.services.llm_service import get_llm
                    self._llm = get_llm()
                except Exception:
                    pass
        return self._llm

    def _call_llm(self, prompt: str) -> str:
        llm = self._get_llm()
        if llm is None:
            return ""
        try:
            response = llm.complete(prompt, temperature=0.7)
            return str(response).strip()
        except Exception as e:
            logger.warning(f"Agent LLM call failed: {e}")
            return ""

    def _load_research_program(self) -> str:
        program_path = os.path.join(
            os.environ.get("GUAARDVARK_ROOT", ""), "data", "research_program.md"
        )
        try:
            with open(program_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "(no research program found)"

    def _format_history(self, history: list, max_rows: int = 20) -> str:
        recent = history[-max_rows:] if len(history) > max_rows else history
        if not recent:
            return "(no experiments yet — this is the first run)"
        lines = ["#  | param | change | delta | status"]
        for i, exp in enumerate(recent, 1):
            param = exp.get("parameter_changed", "?")
            old = exp.get("old_value", "?")
            new = exp.get("new_value", "?")
            delta = exp.get("delta", 0)
            status = exp.get("status", "?")
            delta_str = f"+{delta:.3f}" if delta and delta > 0 else f"{delta:.3f}" if delta else "?"
            lines.append(f"{i}  | {param} | {old}->{new} | {delta_str} | {status}")
        return "\n".join(lines)

    def propose_experiment(
        self, history: list, current_config: dict, phase: int = 1
    ) -> dict:
        """Propose next experiment. Falls back to random if LLM fails."""
        available = [
            p for p in PHASE_PARAMS.get(phase, [])
            if p not in PROTECTED_RAG_PARAMS
        ]
        if not available:
            return self._random_proposal(available or ["top_k"], current_config)

        # Try LLM-driven proposal
        research_program = self._load_research_program()
        history_table = self._format_history(history)
        prompt = AGENT_PROMPT.format(
            current_config=json.dumps(current_config, indent=2),
            phase=phase,
            available_params=", ".join(available),
            research_program=research_program[:2000],
            history_count=min(len(history), 20),
            history_table=history_table,
        )

        response = self._call_llm(prompt)
        try:
            parsed = json.loads(response)
            param = parsed.get("parameter")
            new_value = parsed.get("new_value")
            hypothesis = parsed.get("hypothesis", "LLM-proposed experiment")

            if param in available and new_value is not None:
                # Validate the value is different from current
                if str(new_value) != str(current_config.get(param)):
                    return {
                        "parameter": param,
                        "new_value": new_value,
                        "hypothesis": hypothesis,
                    }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: random proposal
        logger.info("Agent LLM failed to produce valid proposal, falling back to random")
        return self._random_proposal(available, current_config)

    def _random_proposal(self, available: list, current_config: dict) -> dict:
        """Random parameter change as fallback."""
        param = random.choice(available)
        prange = PARAM_RANGES.get(param)
        if not prange:
            return {
                "parameter": param,
                "new_value": not current_config.get(param, False),
                "hypothesis": "Random fallback: toggle boolean",
            }

        low, high, ptype = prange
        current_val = current_config.get(param, low)

        if ptype == "bool":
            new_val = not current_val
        elif ptype == "int":
            new_val = random.randint(int(low), int(high))
            while new_val == current_val and low != high:
                new_val = random.randint(int(low), int(high))
        elif ptype == "float":
            new_val = round(random.uniform(float(low), float(high)), 2)
            while abs(new_val - current_val) < 0.01:
                new_val = round(random.uniform(float(low), float(high)), 2)
        else:
            new_val = current_val

        return {
            "parameter": param,
            "new_value": new_val,
            "hypothesis": f"Random exploration: try {param}={new_val}",
        }

    def should_advance_phase(self, history: list) -> bool:
        """Check if current phase is plateaued."""
        if len(history) < AUTORESEARCH_PHASE_PLATEAU_THRESHOLD:
            return False
        recent = history[-AUTORESEARCH_PHASE_PLATEAU_THRESHOLD:]
        return all(exp.get("status") == "discard" for exp in recent)
```

- [ ] **Step 4: Run tests**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/test_rag_experiment_agent.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/services/rag_experiment_agent.py backend/tests/test_rag_experiment_agent.py data/research_program.md
git commit -m "feat(autoresearch): implement experiment agent with LLM hypothesis engine"
```

---

## Chunk 5: Orchestrator Service

### Task 8: Create the autoresearch orchestrator

**Files:**
- Create: `backend/services/rag_autoresearch_service.py`
- Test: `backend/tests/test_rag_autoresearch_service.py`

- [ ] **Step 1: Write orchestrator tests**

Create `backend/tests/test_rag_autoresearch_service.py`:

```python
"""Tests for the RAG Autoresearch orchestrator."""
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from backend.services.rag_autoresearch_service import RAGAutoresearchService


class TestExperimentCycle:
    def test_single_experiment_keep(self, app):
        """A winning experiment updates the config."""
        with app.app_context():
            svc = RAGAutoresearchService()
            with patch.object(svc.agent, "propose_experiment") as mock_propose, \
                 patch.object(svc.eval_harness, "run_full_eval") as mock_eval, \
                 patch.object(svc, "_load_config") as mock_load, \
                 patch.object(svc, "_save_config") as mock_save, \
                 patch.object(svc, "_log_experiment") as mock_log:
                mock_load.return_value = {
                    "params": {"top_k": 5}, "baseline_score": 3.0, "phase": 1
                }
                mock_propose.return_value = {
                    "parameter": "top_k", "new_value": 8,
                    "hypothesis": "try more chunks",
                }
                mock_eval.return_value = {"composite_score": 3.5, "num_pairs": 10, "details": []}

                result = svc.run_single_experiment()
                assert result["status"] == "keep"
                assert result["delta"] == 0.5
                mock_save.assert_called_once()

    def test_single_experiment_discard(self, app):
        """A losing experiment reverts the config."""
        with app.app_context():
            svc = RAGAutoresearchService()
            with patch.object(svc.agent, "propose_experiment") as mock_propose, \
                 patch.object(svc.eval_harness, "run_full_eval") as mock_eval, \
                 patch.object(svc, "_load_config") as mock_load, \
                 patch.object(svc, "_save_config") as mock_save, \
                 patch.object(svc, "_log_experiment") as mock_log:
                mock_load.return_value = {
                    "params": {"top_k": 5}, "baseline_score": 3.0, "phase": 1
                }
                mock_propose.return_value = {
                    "parameter": "top_k", "new_value": 2,
                    "hypothesis": "try fewer chunks",
                }
                mock_eval.return_value = {"composite_score": 2.5, "num_pairs": 10, "details": []}

                result = svc.run_single_experiment()
                assert result["status"] == "discard"
                assert result["delta"] == -0.5
                mock_save.assert_not_called()  # config not updated on discard


class TestIdleDetection:
    def test_is_idle_returns_true_after_threshold(self):
        """System is idle when last activity exceeds threshold."""
        import time
        svc = RAGAutoresearchService()
        svc._last_activity = time.time() - 700  # 11+ minutes ago
        assert svc.is_idle(idle_minutes=10) is True

    def test_is_idle_returns_false_during_activity(self):
        """System is not idle when recently active."""
        import time
        svc = RAGAutoresearchService()
        svc._last_activity = time.time() - 60  # 1 minute ago
        assert svc.is_idle(idle_minutes=10) is False


class TestPause:
    def test_pause_stops_loop(self):
        """Pause flag prevents next experiment from starting."""
        svc = RAGAutoresearchService()
        svc.pause()
        assert svc._paused is True

    def test_resume_clears_pause(self):
        svc = RAGAutoresearchService()
        svc.pause()
        svc.resume()
        assert svc._paused is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/test_rag_autoresearch_service.py -v
```

- [ ] **Step 3: Implement rag_autoresearch_service.py**

Create `backend/services/rag_autoresearch_service.py`:

```python
"""RAG Autoresearch Orchestrator — the experiment loop.

Coordinates the eval harness, experiment agent, and config management.
Runs experiments when the system is idle, pauses on user activity.
"""
import json
import os
import time
import logging
import uuid
from datetime import datetime
from threading import Lock

from backend.config import (
    AUTORESEARCH_DEFAULT_PARAMS,
    AUTORESEARCH_MAX_EXPERIMENT_DURATION,
)
from backend.services.rag_eval_harness import RAGEvalHarness
from backend.services.rag_experiment_agent import RAGExperimentAgent

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "rag_experiment_config.json"


class RAGAutoresearchService:
    """Core experiment loop orchestrator."""

    def __init__(self):
        self.eval_harness = RAGEvalHarness()
        self.agent = RAGExperimentAgent()
        self._paused = False
        self._running = False
        self._last_activity = time.time()
        self._lock = Lock()
        self._current_experiment_id = None

    # --- Activity tracking ---

    def record_activity(self):
        """Called by activity tracker middleware on user requests."""
        self._last_activity = time.time()
        if self._running:
            self._paused = True

    def is_idle(self, idle_minutes: int = 10) -> bool:
        """Check if system has been idle for the threshold duration."""
        elapsed = time.time() - self._last_activity
        return elapsed > (idle_minutes * 60)

    def is_running(self) -> bool:
        return self._running

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    # --- Config management ---

    def _config_path(self) -> str:
        root = os.environ.get("GUAARDVARK_ROOT", "")
        return os.path.join(root, "data", CONFIG_FILENAME)

    def _load_config(self) -> dict:
        """Load current experiment config from disk."""
        path = self._config_path()
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Initialize with defaults
            config = {
                "version": 1,
                "baseline_score": 0.0,
                "params": dict(AUTORESEARCH_DEFAULT_PARAMS),
                "phase": 1,
                "phase_plateau_count": 0,
            }
            self._save_config(config)
            return config

    def _save_config(self, config: dict):
        """Atomically save config to disk."""
        path = self._config_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(config, f, indent=2)
        os.replace(tmp_path, path)  # atomic on POSIX

    # --- Experiment execution ---

    def run_single_experiment(self) -> dict:
        """Execute one experiment cycle. Returns result dict."""
        config = self._load_config()
        phase = config.get("phase", 1)
        baseline = config.get("baseline_score", 0.0)
        params = config.get("params", dict(AUTORESEARCH_DEFAULT_PARAMS))

        # 1. Get experiment history
        history = self._get_recent_history(limit=20)

        # 2. Check phase transition
        if self.agent.should_advance_phase(history):
            new_phase = min(phase + 1, 3)
            if new_phase != phase:
                logger.info(f"Advancing from Phase {phase} to Phase {new_phase}")
                config["phase"] = new_phase
                config["phase_plateau_count"] = 0
                self._save_config(config)
                phase = new_phase

        # 3. Agent proposes experiment
        proposal = self.agent.propose_experiment(history, params, phase)
        experiment_id = str(uuid.uuid4())
        self._current_experiment_id = experiment_id

        param_name = proposal["parameter"]
        old_value = params.get(param_name)
        new_value = proposal["new_value"]
        hypothesis = proposal.get("hypothesis", "")

        logger.info(
            f"Experiment {experiment_id[:8]}: {param_name} {old_value} -> {new_value} | {hypothesis}"
        )

        # 4. Apply temporary config
        test_params = dict(params)
        test_params[param_name] = new_value

        # 5. Run eval
        t0 = time.time()
        try:
            eval_result = self.eval_harness.run_full_eval(test_params)
            new_score = eval_result["composite_score"]
            duration = time.time() - t0
        except Exception as e:
            logger.error(f"Experiment crashed: {e}")
            result = {
                "experiment_id": experiment_id,
                "parameter": param_name,
                "old_value": str(old_value),
                "new_value": str(new_value),
                "hypothesis": hypothesis,
                "status": "crash",
                "composite_score": 0.0,
                "baseline_score": baseline,
                "delta": 0.0,
                "duration": time.time() - t0,
                "phase": phase,
            }
            self._log_experiment(result)
            return result

        # 6. Compare to baseline
        delta = round(new_score - baseline, 4)
        status = "keep" if new_score > baseline else "discard"

        # 7. Keep or revert
        if status == "keep":
            config["params"][param_name] = new_value
            config["baseline_score"] = new_score
            config["phase_plateau_count"] = 0
            self._save_config(config)
            self._promote_config(config, new_score, "local")
            logger.info(f"KEEP: {param_name}={new_value} score={new_score:.4f} (delta=+{delta:.4f})")
        else:
            config["phase_plateau_count"] = config.get("phase_plateau_count", 0) + 1
            self._save_config(config)
            logger.info(f"DISCARD: {param_name}={new_value} score={new_score:.4f} (delta={delta:.4f})")

        result = {
            "experiment_id": experiment_id,
            "parameter": param_name,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "hypothesis": hypothesis,
            "status": status,
            "composite_score": new_score,
            "baseline_score": baseline,
            "delta": delta,
            "duration": duration,
            "phase": phase,
            "eval_details": eval_result.get("details", []),
        }

        # 8. Log and broadcast
        self._log_experiment(result)
        if status == "keep":
            self._broadcast_to_family(result)
        self._emit_socket_event(result)

        self._current_experiment_id = None
        return result

    def run_loop(self, max_experiments: int = 0):
        """Run experiment loop until paused or max reached."""
        if self._running:
            logger.warning("Autoresearch loop already running")
            return

        self._running = True
        self._paused = False
        count = 0

        try:
            # Check prerequisites
            if not self._check_prerequisites():
                return

            while not self._paused:
                if max_experiments > 0 and count >= max_experiments:
                    logger.info(f"Reached max experiments ({max_experiments})")
                    break

                result = self.run_single_experiment()
                count += 1

                # Check crash protection: 3 consecutive crashes -> stop
                recent = self._get_recent_history(limit=3)
                if len(recent) >= 3 and all(r.get("status") == "crash" for r in recent):
                    logger.error("3 consecutive crashes — pausing autoresearch")
                    break

        finally:
            self._running = False
            self._current_experiment_id = None
            logger.info(f"Autoresearch loop ended after {count} experiments")

    def _check_prerequisites(self) -> bool:
        """Verify system is ready for autoresearch."""
        try:
            from flask import current_app
            if not self.eval_harness.has_sufficient_corpus():
                logger.warning("Insufficient corpus for autoresearch")
                return False
        except RuntimeError:
            pass
        return True

    # --- DB operations ---

    def _log_experiment(self, result: dict):
        """Save experiment result to ExperimentRun table."""
        try:
            from backend.models import ExperimentRun, db
            run = ExperimentRun(
                id=result["experiment_id"],
                phase=result.get("phase", 1),
                parameter_changed=result["parameter"],
                old_value=result.get("old_value"),
                new_value=result.get("new_value"),
                hypothesis=result.get("hypothesis"),
                composite_score=result.get("composite_score", 0.0),
                baseline_score=result.get("baseline_score", 0.0),
                delta=result.get("delta", 0.0),
                status=result["status"],
                eval_details=result.get("eval_details"),
                duration_seconds=result.get("duration"),
            )
            db.session.add(run)
            db.session.commit()
        except Exception as e:
            logger.error(f"Failed to log experiment: {e}")

    def _promote_config(self, config: dict, score: float, source: str):
        """Save winning config to ResearchConfig table."""
        try:
            from backend.models import ResearchConfig, db
            # Deactivate old active config
            ResearchConfig.query.filter_by(is_active=True).update({"is_active": False})
            new_config = ResearchConfig(
                id=str(uuid.uuid4()),
                params=config["params"],
                composite_score=score,
                is_active=True,
                promoted_at=datetime.utcnow(),
                source=source,
            )
            db.session.add(new_config)
            db.session.commit()
        except Exception as e:
            logger.error(f"Failed to promote config: {e}")

    def _get_recent_history(self, limit: int = 20) -> list:
        """Get recent experiment results from DB."""
        try:
            from backend.models import ExperimentRun
            runs = (
                ExperimentRun.query
                .order_by(ExperimentRun.created_at.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in reversed(runs)]
        except Exception:
            return []

    def _broadcast_to_family(self, result: dict):
        """Broadcast winning config via interconnector."""
        try:
            from backend.services.interconnector_sync_service import broadcast_learning
            broadcast_learning(
                learning_type="rag_optimization",
                description=(
                    f"[AUTORESEARCH] {result['parameter']}={result['new_value']}, "
                    f"score={result['composite_score']:.4f}, delta=+{result['delta']:.4f}"
                ),
            )
        except Exception as e:
            logger.debug(f"Family broadcast skipped: {e}")

    def _emit_socket_event(self, result: dict):
        """Emit real-time update via Socket.IO."""
        try:
            from backend.socketio_instance import socketio
            socketio.emit("autoresearch:experiment_complete", {
                "experiment_id": result["experiment_id"],
                "parameter": result["parameter"],
                "status": result["status"],
                "score": result.get("composite_score"),
                "delta": result.get("delta"),
            })
        except Exception:
            pass

    def get_status(self) -> dict:
        """Current status for dashboard."""
        config = self._load_config()
        return {
            "running": self._running,
            "paused": self._paused,
            "current_experiment_id": self._current_experiment_id,
            "phase": config.get("phase", 1),
            "baseline_score": config.get("baseline_score", 0.0),
            "params": config.get("params", {}),
            "total_experiments": self._count_experiments(),
            "total_improvements": self._count_improvements(),
        }

    def _count_experiments(self) -> int:
        try:
            from backend.models import ExperimentRun
            return ExperimentRun.query.count()
        except Exception:
            return 0

    def _count_improvements(self) -> int:
        try:
            from backend.models import ExperimentRun
            return ExperimentRun.query.filter_by(status="keep").count()
        except Exception:
            return 0


# Singleton instance
_autoresearch_service = None


def get_autoresearch_service() -> RAGAutoresearchService:
    global _autoresearch_service
    if _autoresearch_service is None:
        _autoresearch_service = RAGAutoresearchService()
    return _autoresearch_service
```

- [ ] **Step 4: Run tests**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/test_rag_autoresearch_service.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/services/rag_autoresearch_service.py backend/tests/test_rag_autoresearch_service.py
git commit -m "feat(autoresearch): implement orchestrator with keep/revert loop"
```

---

## Chunk 6: API Endpoints & Celery Tasks

### Task 9: Create REST API blueprint

**Files:**
- Create: `backend/api/rag_autoresearch_api.py`
- Modify: `backend/app.py` (register blueprint + activity tracker)

- [ ] **Step 1: Create backend/api/rag_autoresearch_api.py**

```python
"""RAG Autoresearch API — REST endpoints for dashboard and manual triggers."""
from flask import Blueprint, jsonify, request
from backend.services.rag_autoresearch_service import get_autoresearch_service
from backend.models import ExperimentRun, EvalPair, ResearchConfig, Setting, db

autoresearch_bp = Blueprint("autoresearch", __name__, url_prefix="/api/autoresearch")


@autoresearch_bp.route("/status", methods=["GET"])
def get_status():
    svc = get_autoresearch_service()
    return jsonify(svc.get_status())


@autoresearch_bp.route("/start", methods=["POST"])
def start_loop():
    svc = get_autoresearch_service()
    if svc.is_running():
        return jsonify({"error": "Already running"}), 409
    max_exp = request.json.get("max_experiments", 0) if request.is_json else 0
    # Run in background thread
    import threading
    t = threading.Thread(target=svc.run_loop, kwargs={"max_experiments": max_exp}, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@autoresearch_bp.route("/stop", methods=["POST"])
def stop_loop():
    svc = get_autoresearch_service()
    svc.pause()
    return jsonify({"status": "paused"})


@autoresearch_bp.route("/history", methods=["GET"])
def get_history():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    runs = (
        ExperimentRun.query
        .order_by(ExperimentRun.created_at.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )
    return jsonify({
        "experiments": [r.to_dict() for r in runs.items],
        "total": runs.total,
        "page": page,
        "pages": runs.pages,
    })


@autoresearch_bp.route("/config", methods=["GET"])
def get_config():
    svc = get_autoresearch_service()
    config = svc._load_config()
    return jsonify(config)


@autoresearch_bp.route("/config/reset", methods=["POST"])
def reset_config():
    from backend.config import AUTORESEARCH_DEFAULT_PARAMS
    svc = get_autoresearch_service()
    config = {
        "version": 1,
        "baseline_score": 0.0,
        "params": dict(AUTORESEARCH_DEFAULT_PARAMS),
        "phase": 1,
        "phase_plateau_count": 0,
    }
    svc._save_config(config)
    return jsonify({"status": "reset", "config": config})


@autoresearch_bp.route("/eval-pairs", methods=["GET"])
def get_eval_pairs():
    pairs = EvalPair.query.order_by(EvalPair.created_at.desc()).limit(200).all()
    return jsonify({"pairs": [p.to_dict() for p in pairs], "count": len(pairs)})


@autoresearch_bp.route("/eval-pairs/regenerate", methods=["POST"])
def regenerate_eval_pairs():
    svc = get_autoresearch_service()
    pairs = svc.eval_harness.generate_eval_set()
    # Save to DB
    for pair_data in pairs:
        pair = EvalPair(**{k: v for k, v in pair_data.items() if k in EvalPair.__table__.columns.keys()})
        db.session.add(pair)
    db.session.commit()
    return jsonify({"status": "regenerated", "count": len(pairs)})


@autoresearch_bp.route("/settings", methods=["GET"])
def get_settings():
    keys = [
        "rag_autoresearch_idle_minutes",
        "rag_autoresearch_auto_enabled",
        "rag_autoresearch_max_experiments",
        "rag_autoresearch_phase_limit",
        "rag_autoresearch_judge_model",
    ]
    settings = {}
    for key in keys:
        s = Setting.query.filter_by(key=key).first()
        settings[key] = s.value if s else None
    # Apply defaults
    defaults = {
        "rag_autoresearch_idle_minutes": "10",
        "rag_autoresearch_auto_enabled": "true",
        "rag_autoresearch_max_experiments": "0",
        "rag_autoresearch_phase_limit": "2",
        "rag_autoresearch_judge_model": "",
    }
    for k, v in defaults.items():
        if settings[k] is None:
            settings[k] = v
    return jsonify(settings)


@autoresearch_bp.route("/settings", methods=["PUT"])
def update_settings():
    data = request.get_json()
    for key, value in data.items():
        if key.startswith("rag_autoresearch_"):
            s = Setting.query.filter_by(key=key).first()
            if s:
                s.value = str(value)
            else:
                s = Setting(key=key, value=str(value))
                db.session.add(s)
    db.session.commit()
    return jsonify({"status": "updated"})
```

- [ ] **Step 2: Add activity tracker and register blueprint in app.py**

In `backend/app.py`, add the activity tracker near other `before_request` hooks (around line 576):

```python
# Activity tracking for autoresearch idle detection
@app.before_request
def track_user_activity():
    """Update activity timestamp for autoresearch idle detection."""
    from flask import request as req
    # Exempt autoresearch and health endpoints
    if req.path.startswith('/api/autoresearch') or req.path.startswith('/api/health'):
        return
    try:
        from backend.services.rag_autoresearch_service import get_autoresearch_service
        get_autoresearch_service().record_activity()
    except Exception:
        pass
```

Register the blueprint in the blueprint registration section (around line 917):

```python
from backend.api.rag_autoresearch_api import autoresearch_bp
app.register_blueprint(autoresearch_bp)
```

- [ ] **Step 3: Commit**

```bash
git add backend/api/rag_autoresearch_api.py backend/app.py
git commit -m "feat(autoresearch): add REST API endpoints and activity tracker"
```

### Task 10: Create Celery tasks

**Files:**
- Create: `backend/tasks/rag_autoresearch_tasks.py`
- Modify: `backend/celery_app.py` (register tasks, around line 214)

- [ ] **Step 1: Create backend/tasks/rag_autoresearch_tasks.py**

```python
"""Celery tasks for RAG Autoresearch — idle detection, scheduled runs, event triggers."""
import logging
from celery.schedules import crontab

logger = logging.getLogger(__name__)


def create_autoresearch_tasks(celery_app):
    """Create autoresearch Celery tasks."""

    @celery_app.task(name="autoresearch.check_idle")
    def check_idle_and_start():
        """Runs every 60s. Starts autoresearch if system is idle."""
        try:
            from backend.models import Setting
            from backend.services.rag_autoresearch_service import get_autoresearch_service

            svc = get_autoresearch_service()
            if svc.is_running():
                return

            # Read settings
            idle_setting = Setting.query.filter_by(key="rag_autoresearch_idle_minutes").first()
            idle_minutes = int(idle_setting.value) if idle_setting else 10

            auto_setting = Setting.query.filter_by(key="rag_autoresearch_auto_enabled").first()
            auto_enabled = (auto_setting.value.lower() == "true") if auto_setting else True

            if not auto_enabled:
                return

            if svc.is_idle(idle_minutes=idle_minutes):
                max_setting = Setting.query.filter_by(key="rag_autoresearch_max_experiments").first()
                max_exp = int(max_setting.value) if max_setting and max_setting.value != "0" else 0

                logger.info(f"System idle for >{idle_minutes}m — starting autoresearch")
                svc.run_loop(max_experiments=max_exp)
        except Exception as e:
            logger.error(f"Autoresearch idle check failed: {e}")

    @celery_app.task(name="autoresearch.on_index_complete")
    def on_index_complete():
        """Called after indexing completes — marks eval set as potentially stale."""
        try:
            from backend.services.rag_autoresearch_service import get_autoresearch_service
            svc = get_autoresearch_service()
            if svc.eval_harness.is_stale():
                logger.info("Eval set is stale after indexing — will regenerate on next run")
        except Exception as e:
            logger.error(f"Post-index eval check failed: {e}")


def schedule_autoresearch_tasks(celery_app):
    """Register autoresearch Beat schedule."""
    celery_app.conf.beat_schedule.update({
        "autoresearch-idle-check": {
            "task": "autoresearch.check_idle",
            "schedule": 60.0,  # every 60 seconds
        },
    })
```

- [ ] **Step 2: Register in celery_app.py**

Add after the self-improvement task registration (around line 221):

```python
try:
    from backend.tasks.rag_autoresearch_tasks import (
        create_autoresearch_tasks,
        schedule_autoresearch_tasks,
    )
    create_autoresearch_tasks(celery_app)
    schedule_autoresearch_tasks(celery_app)
    logger.info("RAG Autoresearch Celery tasks registered successfully")
except ImportError as e:
    logger.warning(f"Could not import autoresearch tasks: {e}")
```

- [ ] **Step 3: Commit**

```bash
git add backend/tasks/rag_autoresearch_tasks.py backend/celery_app.py
git commit -m "feat(autoresearch): add Celery idle detection and scheduling tasks"
```

---

## Chunk 7: Frontend Dashboard Card

### Task 11: Create RAGAutoresearchCard component

**Files:**
- Create: `frontend/src/components/dashboard/RAGAutoresearchCard.jsx`
- Create: `frontend/src/api/ragAutoresearchService.js`
- Modify: `frontend/src/pages/DashboardPage.jsx` (add card to registry, around line 56)

- [ ] **Step 1: Create frontend/src/api/ragAutoresearchService.js**

```javascript
import apiClient from './apiClient';

export const getAutoresearchStatus = () => apiClient.get('/autoresearch/status');
export const startAutoresearch = (maxExperiments = 0) =>
  apiClient.post('/autoresearch/start', { max_experiments: maxExperiments });
export const stopAutoresearch = () => apiClient.post('/autoresearch/stop');
export const getAutoresearchHistory = (page = 1, perPage = 20) =>
  apiClient.get(`/autoresearch/history?page=${page}&per_page=${perPage}`);
export const getAutoresearchConfig = () => apiClient.get('/autoresearch/config');
export const resetAutoresearchConfig = () => apiClient.post('/autoresearch/config/reset');
export const getAutoresearchSettings = () => apiClient.get('/autoresearch/settings');
export const updateAutoresearchSettings = (settings) =>
  apiClient.put('/autoresearch/settings', settings);
```

- [ ] **Step 2: Create RAGAutoresearchCard.jsx**

Create `frontend/src/components/dashboard/RAGAutoresearchCard.jsx`. This card shows:
- Status indicator (idle/running/paused)
- Current best score and phase
- "Optimize Now" / "Pause" button
- Last 5 experiment results with keep/discard chips
- Total experiments / improvements counters
- Mini score trend (last 20 experiments as a simple bar)

The component polls `/api/autoresearch/status` every 5 seconds when running, and listens for Socket.IO `autoresearch:experiment_complete` events for real-time updates.

Key implementation notes:
- Use MUI components: `Chip`, `IconButton`, `LinearProgress`, `Tooltip`
- Follow the pattern of `FamilySelfImprovementCard.jsx` for card structure
- Use `DashboardCardWrapper` as the outer wrapper
- Socket.IO listener for live updates during experiments

```jsx
import React, { useState, useEffect, useCallback } from 'react';
import {
  Box, Typography, Button, Chip, LinearProgress, Tooltip,
  Table, TableBody, TableRow, TableCell,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Science as ScienceIcon,
  TrendingUp as TrendingUpIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import {
  getAutoresearchStatus,
  startAutoresearch,
  stopAutoresearch,
  getAutoresearchHistory,
} from '../../api/ragAutoresearchService';
import io from 'socket.io-client';

const RAGAutoresearchCard = () => {
  const [status, setStatus] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await getAutoresearchStatus();
      setStatus(res.data);
    } catch (e) { /* backend may not have endpoint yet */ }
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await getAutoresearchHistory(1, 5);
      setHistory(res.data.experiments || []);
    } catch (e) { /* ignore */ }
  }, []);

  useEffect(() => {
    fetchStatus();
    fetchHistory();
    const interval = setInterval(() => {
      fetchStatus();
      if (status?.running) fetchHistory();
    }, 5000);
    return () => clearInterval(interval);
  }, [status?.running]);

  const handleStart = async () => {
    setLoading(true);
    try {
      await startAutoresearch();
      await fetchStatus();
    } finally { setLoading(false); }
  };

  const handleStop = async () => {
    await stopAutoresearch();
    await fetchStatus();
  };

  if (!status) {
    return (
      <Box sx={{ p: 1, textAlign: 'center' }}>
        <Typography variant="body2" color="text.secondary">Loading...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 1, height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Status row */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <ScienceIcon sx={{ fontSize: 16, color: status.running ? 'success.main' : 'text.secondary' }} />
          <Chip
            label={status.running ? 'Running' : status.paused ? 'Paused' : 'Idle'}
            size="small"
            color={status.running ? 'success' : 'default'}
            sx={{ height: 20, fontSize: '0.7rem' }}
          />
          <Typography variant="caption" color="text.secondary">
            Phase {status.phase}
          </Typography>
        </Box>
        {status.running ? (
          <Tooltip title="Pause optimization">
            <IconButton size="small" onClick={handleStop}><PauseIcon sx={{ fontSize: 16 }} /></IconButton>
          </Tooltip>
        ) : (
          <Tooltip title="Start optimization">
            <IconButton size="small" onClick={handleStart} disabled={loading}>
              <PlayIcon sx={{ fontSize: 16 }} />
            </IconButton>
          </Tooltip>
        )}
      </Box>

      {status.running && <LinearProgress sx={{ mb: 1, borderRadius: 1 }} />}

      {/* Score */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="caption">
          Score: <strong>{status.baseline_score?.toFixed(3) || '—'}</strong>
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {status.total_experiments} runs / {status.total_improvements} improvements
        </Typography>
      </Box>

      {/* Recent experiments */}
      {history.length > 0 && (
        <Box sx={{ flex: 1, overflow: 'auto' }}>
          <Table size="small" sx={{ '& td': { py: 0.25, px: 0.5, fontSize: '0.7rem' } }}>
            <TableBody>
              {history.map((exp) => (
                <TableRow key={exp.id}>
                  <TableCell>{exp.parameter_changed}</TableCell>
                  <TableCell>{exp.new_value}</TableCell>
                  <TableCell>
                    <Chip
                      label={exp.status}
                      size="small"
                      color={exp.status === 'keep' ? 'success' : exp.status === 'crash' ? 'error' : 'default'}
                      sx={{ height: 16, fontSize: '0.6rem' }}
                    />
                  </TableCell>
                  <TableCell align="right">
                    {exp.delta > 0 ? `+${exp.delta.toFixed(3)}` : exp.delta?.toFixed(3)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Box>
      )}

      {history.length === 0 && !status.running && (
        <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography variant="caption" color="text.secondary">
            No experiments yet. Click play to start.
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default RAGAutoresearchCard;
```

- [ ] **Step 3: Register card in DashboardPage.jsx**

Add import (around line 42):
```javascript
import RAGAutoresearchCard from "../components/dashboard/RAGAutoresearchCard";
```

Add to cardComponents registry (around line 56):
```javascript
autoresearch: RAGAutoresearchCard,
```

- [ ] **Step 4: Build frontend to verify**

```bash
cd /home/llamax1/LLAMAX7/frontend && npm run build
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/dashboard/RAGAutoresearchCard.jsx frontend/src/api/ragAutoresearchService.js frontend/src/pages/DashboardPage.jsx
git commit -m "feat(autoresearch): add dashboard card with experiment history and controls"
```

---

## Chunk 8: Settings UI & Integration

### Task 12: Add autoresearch settings to SettingsPage

**Files:**
- Modify: `frontend/src/pages/SettingsPage.jsx` (A.I. section, after embedding model dropdown around line 2362)

- [ ] **Step 1: Add autoresearch settings section**

After the existing embedding model section in the A.I. tab, add a new subsection with:
- Idle threshold slider (5-120 minutes, default 10)
- Auto-optimization toggle (enable/disable idle trigger)
- Phase limit selector (1/2/3)
- "Reset to Defaults" button for RAG config

Use the existing `getAutoresearchSettings` / `updateAutoresearchSettings` API calls. Follow the pattern of existing settings (GPU resources section, embedding model section) for consistent styling.

- [ ] **Step 2: Build and verify**

```bash
cd /home/llamax1/LLAMAX7/frontend && npm run build
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/SettingsPage.jsx
git commit -m "feat(autoresearch): add optimization settings to A.I. settings section"
```

### Task 13: Wire indexing post-hook for eval staleness

**Files:**
- Modify: `backend/services/indexing_service.py` (after successful indexing completes)

- [ ] **Step 1: Add post-index hook**

Find the success path in the indexing functions (where documents are successfully indexed) and add:

```python
# Notify autoresearch that corpus has changed
try:
    from backend.tasks.rag_autoresearch_tasks import on_index_complete
    on_index_complete.delay()
except Exception:
    pass  # autoresearch is optional
```

- [ ] **Step 2: Commit**

```bash
git add backend/services/indexing_service.py
git commit -m "feat(autoresearch): add post-indexing hook for eval staleness detection"
```

---

## Chunk 9: Integration Testing & Polish

### Task 14: End-to-end integration test

**Files:**
- Create: `backend/tests/test_autoresearch_integration.py`

- [ ] **Step 1: Write integration test**

```python
"""End-to-end test for the autoresearch experiment loop."""
import pytest
from unittest.mock import patch, MagicMock
from backend.services.rag_autoresearch_service import RAGAutoresearchService


class TestEndToEnd:
    def test_full_experiment_cycle(self, app):
        """Complete cycle: propose -> eval -> keep/discard -> log."""
        with app.app_context():
            svc = RAGAutoresearchService()

            # Mock the LLM calls (eval harness + agent)
            with patch.object(svc.eval_harness, "_call_llm") as mock_llm, \
                 patch.object(svc.agent, "_call_llm") as mock_agent_llm, \
                 patch.object(svc.eval_harness, "_get_active_eval_pairs") as mock_pairs, \
                 patch.object(svc.eval_harness, "has_sufficient_corpus", return_value=True):

                # Setup eval pairs
                mock_pairs.return_value = [
                    {"id": "p1", "question": "q1", "expected_answer": "a1"},
                ]

                # Agent proposes top_k=8
                mock_agent_llm.return_value = (
                    '{"parameter": "top_k", "new_value": 8, "hypothesis": "test"}'
                )

                # Eval harness scores (judge + response generation)
                mock_llm.side_effect = [
                    "The answer based on context.",  # response generation
                    '{"relevance": 4, "grounding": 4, "completeness": 4}',  # judge
                ]

                result = svc.run_single_experiment()
                assert result["status"] in ("keep", "discard")
                assert result["parameter"] == "top_k"
                assert "composite_score" in result

    def test_loop_respects_pause(self, app):
        """Loop stops when paused."""
        with app.app_context():
            svc = RAGAutoresearchService()
            svc._paused = True  # pre-pause

            with patch.object(svc, "_check_prerequisites", return_value=True):
                svc.run_loop(max_experiments=5)
                # Should exit immediately without running experiments
                assert not svc.is_running()
```

- [ ] **Step 2: Run all autoresearch tests**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/test_autoresearch*.py backend/tests/test_rag_*.py backend/tests/test_experiment_context*.py -v
```

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_autoresearch_integration.py
git commit -m "test(autoresearch): add end-to-end integration tests"
```

### Task 15: Final build verification

- [ ] **Step 1: Run full backend test suite**

```bash
cd /home/llamax1/LLAMAX7 && python3 -m pytest backend/tests/ -v --timeout=60 2>&1 | tail -30
```

- [ ] **Step 2: Build frontend**

```bash
cd /home/llamax1/LLAMAX7/frontend && npm run build
```

- [ ] **Step 3: Generate migration if not already done**

```bash
cd /home/llamax1/LLAMAX7/backend && source venv/bin/activate && python3 -c "from backend.models import ExperimentRun, EvalPair, ResearchConfig; print('Models import OK')"
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat(autoresearch): complete RAG autoresearch v1 implementation"
```

---

## Summary

| Chunk | Tasks | Description |
|---|---|---|
| 1 | 1 | DB models + migration |
| 2 | 2-4 | Config constants + experiment context plumbing |
| 3 | 5 | Eval harness (the "prepare.py") |
| 4 | 6-7 | Research program + experiment agent |
| 5 | 8 | Orchestrator service (the loop) |
| 6 | 9-10 | API endpoints + Celery tasks |
| 7 | 11 | Frontend dashboard card |
| 8 | 12-13 | Settings UI + indexing hook |
| 9 | 14-15 | Integration tests + verification |

**Total: 15 tasks across 9 chunks. Estimated new files: 10. Modified files: 7.**

**What's deferred to Phase 2 (family architecture extension):**
- Uncle Claude experiment log review (enhance existing 12h Celery task)
- `review_config_change()` method on `claude_advisor_service.py`
- Shadow corpus indexing for Phase 2 experiments
- Interconnector family suggestion queue + dedup
- Score-over-time chart in dashboard card
- Phase 3 (embedding model switching) integration
