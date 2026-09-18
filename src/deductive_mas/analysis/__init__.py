"""Static + dynamic program analysis. Knows nothing about misconceptions, that
is wired together in pipeline.py.
"""

from .align import Alignment, classify_divergence, locate_divergence, needleman_wunsch, trajectory
from .ast_features import CodeFeatures, extract
from .belief import FusionResult, MassFunction, fuse
from .cfg import ControlFlowGraph, build_cfg, synchronisation_lines
from .complexity import Asymptotic, ComplexityReport, estimate
from .counterexample import CounterexampleSearch, SearchOutcome, shrink
from .dataflow import DataflowFacts, analyse as solve_dataflow
from .leakage import LeakageDetector, LeakageReport, fingerprint, winnow
from .normalize import NormalForm, normalise, structurally_equivalent
from .tracer import ExecutionResult, GuardedRunner, TraceEvent
from .varmatch import RoleMapping, match_variables
