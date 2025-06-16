from __future__ import annotations

import google_benchmark
import misc_benchmark  # noqa

# explicit imports to register all benchmarks
import reducer_benchmark  # noqa

if __name__ == "__main__":
    google_benchmark.main()
