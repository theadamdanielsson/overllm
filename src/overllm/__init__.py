"""overllm - catch the LLM/AI calls you didn't need.

A fast, deterministic (no-LLM) linter that flags LLM API calls where plain,
cheaper, more reliable code would do the same job. Built on Python's own `ast`.
"""

__version__ = "0.1.2"

from .models import Finding

__all__ = ["Finding", "__version__"]
