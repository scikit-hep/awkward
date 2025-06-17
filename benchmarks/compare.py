from __future__ import annotations

import json
import os
import pathlib
import sys


def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)


def relative_difference(val1, val2):
    return abs(val1 - val2) / min(val1, val2)


def format_benchmark_name(name: str) -> str:
    try:
        parts = name.split("/")
        base = parts[0]
        params = dict(part.split("=", 1) for part in parts[1:])

        array = params.pop("array", "??")
        length = params.pop("length", "??")
        dtype = params.pop("dtype", "").replace("'", "")
        dtype_short = {
            "float64": "f64",
            "float32": "f32",
            "int64": "i64",
            "int32": "i32",
        }.get(dtype, dtype)

        pretty_name = f"{base}({array}<{length},{dtype_short}>"

        # any extra parameters to the function, e.g. `axis=0`
        for k, v in params.items():
            pretty_name += f", {k}={v}"

        pretty_name += ")"
        return pretty_name
    except Exception:
        return name  # fallback


def compare_benchmarks(
    file1_path,
    file2_path,
    output_path,
    threshold=0.1,
):
    data1 = load_json(file1_path)
    data2 = load_json(file2_path)

    bm1 = {b["name"]: b for b in data1["benchmarks"]}
    bm2 = {b["name"]: b for b in data2["benchmarks"]}

    output_lines = []

    found_diffs = False

    for name in bm1:  # noqa: PLC0206
        if name in bm2:
            b1 = bm1[name]
            b2 = bm2[name]

            cpu1 = b1["cpu_time"]
            cpu2 = b2["cpu_time"]
            rel_diff = relative_difference(cpu1, cpu2)

            if rel_diff > threshold:
                found_diffs = True
                display_name = format_benchmark_name(name)

                direction = "🟢 **Improvement**" if cpu2 < cpu1 else "🔴 **Regression**"
                diff_line = f"**Relative CPU Time Difference:** `{rel_diff * 100:.1f}%` — {direction}"

                output_lines.append(f"### 🔹 {display_name}\n")
                output_lines.append(diff_line + "\n")

                # Collapsed detailed comparison
                output_lines.append(
                    "<details><summary>Show full comparison</summary>\n\n"
                )

                def _parse_branch_sha(file_path):
                    assert file_path.endswith(".json")
                    file_path = file_path.replace(".json", "") # remove file ending
                    file_path = file_path.replace(os.getenv("BASE_OUTPUT_DIR", "results") + "/", "", 1) # remove base dir
                    branch, sha = file_path.rsplit("__", 1) # get branch & SHA
                    return f"branch: `{branch}` (sha: {sha})"

                headers = [
                    "Metric",
                    _parse_branch_sha(file1_path),
                    _parse_branch_sha(file2_path),
                ]
                time_unit = b1.get("time_unit", "")
                assert time_unit == b2.get("time_unit", ""), (
                    "Can't compare difference units"
                )

                table = [
                    "| " + " | ".join(headers) + " |",
                    "| --- | --- | --- |",
                    f"| `cpu_time` ({time_unit}) | {cpu1:.6e} | {cpu2:.6e} |",
                    f"| `real_time` ({time_unit}) | {b1['real_time']:.6e} | {b2['real_time']:.6e} |",
                ]
                eps = "elements_per_second"  # potential extra info
                if eps in b1 and eps in b2:
                    table += [f"| `elements/s` (Hz) | {b1[eps]:.2e} | {b2[eps]:.2e} |"]

                output_lines.extend(table)
                output_lines.append("\n</details>\n")

    if not found_diffs:
        print(  # noqa: T201
            f"No significant differences (over {threshold * 100:.1f}%) in cpu_time found."
        )
        return

    header = ["## Benchmarks"]
    markdown_output = "\n".join(header + output_lines)

    # Print to terminal
    print(markdown_output)  # noqa: T201

    # Save to file
    with open(output_file, "w") as f:
        f.write(markdown_output + "\n")

    print(f"\n✅ Detailed Markdown saved to `{output_file}`")  # noqa: T201


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(__file__)} file1.json file2.json")  # noqa: T201
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    output_file = str(
        pathlib.Path(os.getenv("BASE_OUTPUT_DIR", "results"))
        / pathlib.Path("comparison.md")
    )
    compare_benchmarks(file1, file2, output_file)
