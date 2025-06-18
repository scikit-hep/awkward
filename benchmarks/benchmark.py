from __future__ import annotations

import google_benchmark

# explicit imports to register all benchmarks
import misc_benchmark  # noqa: F401
import reducer_benchmark  # noqa: F401

if __name__ == "__main__":
    google_benchmark.main()
