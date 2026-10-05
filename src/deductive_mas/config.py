"""Config dataclasses. Values can be set in code, loaded from a json file or
overridden with DMAS_* env vars. Order: explicit arg > env > file > default.
"""

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .errors import ConfigurationError

_ENV_PREFIX = "DMAS_"

# the paper's live runs went through groq's openai-compatible endpoint; the
# claude backend has its own host, see llm/claude.py
GROQ_URL = "https://api.groq.com/openai/v1"
ANTHROPIC_URL = "https://api.anthropic.com"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(_ENV_PREFIX + name, default)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{_ENV_PREFIX}{name} must be a float, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{_ENV_PREFIX}{name} must be an int, got {raw!r}") from exc


@dataclass(frozen=True)
class ExecutionConfig:
    """Limits for the traced interpreter. Not a real sandbox, just enough for
    classroom code.
    """

    max_steps: int = 40_000
    max_seconds: float = 2.0
    max_recursion: int = 400
    max_trace_events: int = 2_000
    # how similar a matched variable has to behave before we quote its values
    # to the student. Below this the role is dropped from the comparison; if
    # none survive we only compare outputs (happens when the student used a
    # different algorithm).
    min_role_confidence: float = 0.72
    max_container_preview: int = 12
    allowed_imports: tuple = (
        "math", "collections", "heapq", "bisect", "itertools", "functools", "typing",
    )
    # the counterexample search runs the code hundreds of times so it gets a
    # much smaller budget. Going over the budget is itself a finding.
    search_max_steps: int = 12_000
    search_max_seconds: float = 0.4
    search_random_attempts: int = 80


@dataclass(frozen=True)
class DiagnosisConfig:
    """Thresholds for the evaluator's belief fusion."""

    # min fused belief to report a misconception at all
    report_threshold: float = 0.22
    # min belief for the top hit to count as a settled diagnosis. Between this
    # and report_threshold we still show it but mark the session inconclusive.
    primary_threshold: float = 0.40
    # reliability of each evidence source (discounting in Dempster fusion)
    static_reliability: float = 0.86
    dynamic_reliability: float = 0.92
    # model evidence is the weakest on purpose
    llm_reliability: float = 0.62
    max_reported: int = 5
    # mass that a clean differential test puts on "no correctness bug".
    # Not 1.0 because we only sample the inputs.
    dynamic_refutation: float = 0.55
    # evidence sources to use, drop one for the ablation
    sources: tuple = ("symbolic", "dynamic", "cost", "model")
    # dempster (default) or max / mean / noisy-or for the ablation
    fusion_rule: str = "dempster"
    # the narrative is never scored so benchmarks can skip it (one less model call)
    narrate: bool = True


@dataclass(frozen=True)
class KnowledgeConfig:
    """Blame propagation and mastery (BKT) parameters."""

    # damping for personalised PageRank on the reversed prerequisite DAG
    blame_damping: float = 0.62
    blame_iterations: int = 64
    blame_tolerance: float = 1e-10
    # below this posterior a concept counts as not mastered
    mastery_threshold: float = 0.62
    # BKT defaults, can be overridden per concept
    bkt_prior: float = 0.30
    bkt_learn: float = 0.20
    bkt_slip: float = 0.10
    bkt_guess: float = 0.20
    # a concept needs at least this fraction of the max blame to be a target,
    # otherwise PageRank leaks a bit of mass everywhere and some far away
    # foundation concept could win.
    min_blame_ratio: float = 0.15
    # how much a weak prerequisite pulls down the concept that depends on it
    prerequisite_coupling: float = 0.55


@dataclass(frozen=True)
class RetrievalConfig:
    """Hybrid retrieval params (BM25 + LSA)."""

    bm25_k1: float = 1.35
    bm25_b: float = 0.72
    lsa_components: int = 24
    lsa_power_iterations: int = 3
    lsa_oversampling: int = 8
    # RRF constant. 60 is the usual value but that is for thousands of docs,
    # with ~50 cards it flattens the top of the ranking, so use something
    # closer to the result set size.
    rrf_k: float = 12.0
    mmr_lambda: float = 0.84
    top_k: int = 6
    candidate_pool: int = 24
    # min support for one assertive sentence. A made up but on-topic claim gets
    # ~0.35, a paraphrase of a card gets ~0.5+.
    min_grounding_score: float = 0.40


@dataclass(frozen=True)
class InterventionConfig:
    """ZPD and leakage settings."""

    # success probability band = ZPD
    zpd_low: float = 0.45
    zpd_high: float = 0.85
    hint_levels: int = 3
    # winnowing params for the MOSS style leak detector
    fingerprint_kgram: int = 5
    fingerprint_window: int = 4
    # reject hints whose containment vs the reference solution is above this
    max_leakage: float = 0.18
    diversity_lambda: float = 0.65


@dataclass(frozen=True)
class LLMConfig:
    """Which LLM backend to use.

    The defaults are the paper's setup: openai/gpt-oss-120b through groq's
    OpenAI-compatible endpoint. backend='auto' uses that if there is an api
    key (OPENAI_API_KEY or DMAS_API_KEY), otherwise the offline reasoner, so
    it always runs. backend='openai' works with any OpenAI-compatible chat
    completions server (e.g. a local model). backend='claude' reads
    ANTHROPIC_API_KEY and needs a claude model name.
    """

    backend: str = "auto"  # auto | claude | openai | offline
    model: str = "openai/gpt-oss-120b"
    base_url: str = GROQ_URL
    api_key: Optional[str] = None
    # Claude reasoning effort: low | medium | high | xhigh | max
    effort: str = "medium"
    # only sent to OpenAI-compatible endpoints (Claude rejects sampling params)
    temperature: float = 0.2
    max_tokens: int = 4000
    timeout: float = 120.0
    max_retries: int = 6
    # longest single sleep when the server tells us to wait
    max_wait: float = 90.0
    cache_dir: Optional[str] = ".dmas_cache"

    def resolved_key(self, kind: str = "claude") -> Optional[str]:
        """Api key for kind ('claude' or 'openai') if we have one."""
        if self.api_key:
            return self.api_key
        if kind == "claude":
            return os.environ.get("ANTHROPIC_API_KEY") or _env("API_KEY")
        return os.environ.get("OPENAI_API_KEY") or _env("API_KEY")


@dataclass(frozen=True)
class Config:
    """Top level config."""

    seed: int = 20260909
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    diagnosis: DiagnosisConfig = field(default_factory=DiagnosisConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    intervention: InterventionConfig = field(default_factory=InterventionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    color: str = "auto"  # auto | always | never

    # loading
    @classmethod
    def from_env(cls) -> "Config":
        base = cls()
        llm = replace(
            base.llm,
            backend=_env("BACKEND", base.llm.backend) or base.llm.backend,
            model=_env("MODEL", base.llm.model) or base.llm.model,
            base_url=_env("BASE_URL", base.llm.base_url) or base.llm.base_url,
            temperature=_env_float("TEMPERATURE", base.llm.temperature),
            effort=_env("EFFORT", base.llm.effort) or base.llm.effort,
        )
        return replace(
            base,
            seed=_env_int("SEED", base.seed),
            color=_env("COLOR", base.color) or base.color,
            llm=llm,
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        cfg = cls.from_env()
        if path is None:
            return cfg
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cfg.merged(payload)

    def merged(self, payload: Mapping[str, Any]) -> "Config":
        """Copy of this config with the nested dict payload applied on top."""
        groups: Dict[str, Any] = {}
        scalars: Dict[str, Any] = {}
        for key, value in payload.items():
            current = getattr(self, key, None)
            if current is None and not hasattr(self, key):
                raise ConfigurationError(f"unknown configuration key: {key!r}")
            if isinstance(value, Mapping):
                groups[key] = replace(current, **dict(value))
            else:
                scalars[key] = value
        return replace(self, **{**scalars, **groups})


DEFAULT_CONFIG = Config()
