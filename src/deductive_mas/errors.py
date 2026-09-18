"""All the exceptions used in the package. Most of them get caught by the
orchestrator so a session degrades instead of crashing.
"""

class DMASError(Exception):
    """Base class for everything below."""


class ConfigurationError(DMASError):
    """Bad or missing config value."""


class AnalysisError(DMASError):
    """Static or dynamic analysis could not run."""


class UnsafeCodeError(AnalysisError):
    """The submission uses something the guarded runner does not allow."""


class ExecutionTimeout(AnalysisError):
    """Traced execution went over the step / time budget."""


class KnowledgeGraphError(DMASError):
    """Broken ontology (unknown concept, cycle in prerequisites...)."""


class RetrievalError(DMASError):
    """Retrieval index is empty or broken."""


class BackendError(DMASError):
    """The LLM backend failed or returned garbage."""

    def __init__(self, message: str, *, status: int = 0):
        super().__init__(message)
        # http status if there was one
        self.status = status


class VerificationRejected(DMASError):
    """Generated hints kept failing the gate."""
