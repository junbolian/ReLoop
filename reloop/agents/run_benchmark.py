"""
Compatibility shim that forwards to the new CLI module.
"""

import warnings

from .cli.run_benchmark import main

if __name__ == "__main__":  # pragma: no cover
    warnings.warn(
        "reloop.agents.run_benchmark is deprecated; use python -m reloop.agents.cli.run_benchmark instead.",
        DeprecationWarning,
    )
    main()
