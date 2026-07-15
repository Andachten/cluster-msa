class ClusterMsaError(Exception):
    """Base exception for cluster-msa failures."""


class InputValidationError(ClusterMsaError):
    """Raised when an input sequence file is invalid."""


class ConfigurationError(ClusterMsaError):
    """Raised when runtime configuration is invalid."""


class ExternalToolError(ClusterMsaError):
    """Raised when an external tool cannot complete its work."""


class OutputValidationError(ClusterMsaError):
    """Raised when generated output is missing or invalid."""
