#!/usr/bin/env python
"""One-command Kintsugi demo.

This is a thin wrapper around the package-internal demo so that repo
users can still run ``python examples/demo.py``.  For pip-installed
users the canonical invocations are::

    kintsugi-demo           # CLI entry point
    python -m kintsugi.demo # module invocation
"""

from __future__ import annotations

from kintsugi._demo import main

if __name__ == "__main__":
    main()
