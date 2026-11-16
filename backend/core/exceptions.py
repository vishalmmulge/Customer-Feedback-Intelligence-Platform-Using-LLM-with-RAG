"""Custom application exceptions.

Every service error is typed so the FastAPI exception handlers can turn them
into consistent, machine-readable JSON responses instead of 500 HTML pages.
"""

from typing import Any, Dict, Optional


class AppError(Exception):
    """Base application error with an error code, HTTP status, and details."""

    def __init__(
        self,
        message: str,
        code: str = "error",
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ValidationError(AppError):
    """Raised when input fails business validation (400)."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="validation_error", status_code=400, details=details)


class NotFoundError(AppError):
    """Raised when a requested resource does not exist (404)."""

    def __init__(self, message: str):
        super().__init__(message, code="not_found", status_code=404)


class ProviderError(AppError):
    """Raised when an external provider (Ollama, FAISS) fails (502)."""

    def __init__(self, message: str):
        super().__init__(message, code="provider_error", status_code=502)
