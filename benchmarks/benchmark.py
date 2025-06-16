import google_benchmark

# explicit imports to register all benchmarks
import reducer_benchmark  # noqa
import misc_benchmark  # noqa


if __name__ == "__main__":
    google_benchmark.main()
