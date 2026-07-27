"""Compare CPU kernel performance: released `pip install awkward` vs the local
checkout (this branch), using benchmarks/bench_ak_funcs_cpu.py for both.

Both runs happen on the SAME machine, back to back, so the timings are
comparable. The released build runs in a throwaway virtualenv; the branch
runs in the interpreter you launch this with (so launch it from your branch
dev environment, where `awkward` + the locally built `awkward-cpp` import).

Usage
-----
    # from the branch dev env (the one where `import awkward` is the checkout):
    python benchmarks/compare_pip_vs_branch.py
    python benchmarks/compare_pip_vs_branch.py --pip-version 2.9.1
    python benchmarks/compare_pip_vs_branch.py --only sort_axis1 argsort_axis1

It prints a table of best-of-N milliseconds for pip vs branch and the speedup
(pip / branch; >1.0 means the branch is faster). Raw JSON for each run is kept
so you can re-diff without re-running.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import venv

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench_ak_funcs_cpu.py")


def run_bench(python_exe: str, out_json: str, env: dict | None = None) -> dict:
    subprocess.run([python_exe, BENCH, out_json], check=True, env=env)
    with open(out_json) as f:
        return json.load(f)


def run_pip_baseline(out_json: str, pip_spec: str) -> dict:
    """Build a throwaway venv with the released awkward and run the bench in it."""
    vdir = tempfile.mkdtemp(prefix="ak_pip_bench_")
    venv.create(vdir, with_pip=True)
    py = os.path.join(vdir, "bin", "python")
    if not os.path.exists(py):  # Windows
        py = os.path.join(vdir, "Scripts", "python.exe")
    subprocess.run([py, "-m", "pip", "install", "-q", "numpy", pip_spec], check=True)
    return run_bench(py, out_json)


def get_ms(v):
    """Accept both the new {'ms','peak_mb'} dicts and old bare-float JSONs."""
    return v.get("ms") if isinstance(v, dict) else v


def get_mb(v):
    return v.get("peak_mb") if isinstance(v, dict) else None


def fmt_ms(ms: float | None) -> str:
    return f"{ms * 1e3:9.3f}" if ms is not None else f"{'n/a':>9}"


def fmt_mb(mb: float | None) -> str:
    return f"{mb:8.2f}" if mb is not None else f"{'n/a':>8}"


def geomean(xs: list[float]) -> float:
    g = 1.0
    for x in xs:
        g *= x
    return g ** (1.0 / len(xs)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pip-version",
        default=None,
        help="pin the released awkward (e.g. 2.9.1); default: latest",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="restrict the printed table to these op names",
    )
    ap.add_argument("--pip-json", default=None, help="reuse an existing pip-run JSON")
    ap.add_argument(
        "--branch-json", default=None, help="reuse an existing branch-run JSON"
    )
    args = ap.parse_args()

    pip_spec = "awkward" + (f"=={args.pip_version}" if args.pip_version else "")
    tmp = tempfile.mkdtemp(prefix="ak_bench_cmp_")

    print("== branch run (this interpreter) ==", flush=True)
    branch_json = args.branch_json or os.path.join(tmp, "branch.json")
    branch = (
        json.load(open(branch_json))
        if args.branch_json
        else run_bench(sys.executable, branch_json)
    )

    print(f"== pip run ({pip_spec}, isolated venv) ==", flush=True)
    pip_json = args.pip_json or os.path.join(tmp, "pip.json")
    pip = (
        json.load(open(pip_json))
        if args.pip_json
        else run_pip_baseline(pip_json, pip_spec)
    )

    keys = sorted(set(pip) | set(branch))
    header = (
        f"\n{'scale|op|dtype':46s} {'pip ms':>9} {'br ms':>9} {'t x':>7}  "
        f"{'pip MB':>8} {'br MB':>8} {'mem x':>7}"
    )
    print(header)
    print("-" * len(header))
    speedups, memratios = [], []
    for k in keys:
        if args.only and not any(o in k for o in args.only):
            continue
        pv, bv = pip.get(k), branch.get(k)
        pms, bms = get_ms(pv), get_ms(bv)
        pmb, bmb = get_mb(pv), get_mb(bv)
        tx = f"{pms / bms:6.2f}x" if (pms and bms) else f"{'-':>7}"
        mx = f"{pmb / bmb:6.2f}x" if (pmb and bmb) else f"{'-':>7}"
        if pms and bms:
            speedups.append(pms / bms)
        if pmb and bmb:
            memratios.append(pmb / bmb)
        print(
            f"{k:46s} {fmt_ms(pms)} {fmt_ms(bms)} {tx:>7}  "
            f"{fmt_mb(pmb)} {fmt_mb(bmb)} {mx:>7}"
        )

    print("-" * len(header))
    if speedups:
        speedups.sort()
        print(
            f"time : geomean {geomean(speedups):.3f}x  "
            f"min {speedups[0]:.2f}x  max {speedups[-1]:.2f}x   (>1 = branch faster)"
        )
    if memratios:
        memratios.sort()
        print(
            f"mem  : geomean {geomean(memratios):.3f}x  "
            f"min {memratios[0]:.2f}x  max {memratios[-1]:.2f}x   (>1 = branch leaner)"
        )
    elif not any(get_mb(v) is not None for v in branch.values()):
        print("mem  : no peak_mb in the JSONs (re-run the bench with psutil installed)")
    print(f"\nJSON: pip={pip_json}  branch={branch_json}")


if __name__ == "__main__":
    main()
