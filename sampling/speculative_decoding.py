"""
speculative_decoding.py
向后兼容的 re-export 入口。

inference.py 中的导入路径无需任何修改：
  from sampling.speculative_decoding import (
      speculative_generate,
      speculative_generate_pregeneration,
      speculative_generate_tree_pregen,
  )
"""

from .speculative_standard      import speculative_generate                  # noqa: F401
from .speculative_pregeneration import speculative_generate_pregeneration    # noqa: F401
from .speculative_tree          import speculative_generate_tree_pregen      # noqa: F401
from .speculative_utils         import max_fn, make_tree_stats               # noqa: F401

__all__ = [
    "speculative_generate",
    "speculative_generate_pregeneration",
    "speculative_generate_tree_pregen",
    "max_fn",
    "make_tree_stats",
]