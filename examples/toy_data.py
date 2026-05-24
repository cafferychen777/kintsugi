"""Generate a small synthetic spatial transcriptomics dataset.

This is a thin wrapper around ``kintsugi._demo.make_toy_dataset`` so
that repository-level scripts can still do::

    from examples.toy_data import make_toy_dataset

For pip-installed users the canonical path is ``kintsugi.make_toy_dataset``.
"""

from __future__ import annotations

from kintsugi._demo import make_toy_dataset

__all__ = ["make_toy_dataset"]

if __name__ == "__main__":
    from kintsugi._demo import main
    main()
