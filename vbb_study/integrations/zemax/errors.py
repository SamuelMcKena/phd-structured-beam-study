"""Typed exceptions for the optional Zemax/OpticStudio bridge."""

class ZemaxIntegrationError(RuntimeError):
    """Base error for a diagnosed Zemax integration failure."""

class ZemaxUnavailableError(ZemaxIntegrationError):
    """Raised when the optional OpticStudio integration is unavailable."""

class ZemaxModelError(ZemaxIntegrationError):
    """Raised when a supplied prescription is invalid or incompatible."""

class ZemaxFieldExchangeError(ZemaxIntegrationError):
    """Raised when a requested field exchange is unsupported or invalid."""
