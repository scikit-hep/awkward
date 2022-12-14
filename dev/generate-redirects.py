import argparse
import json
import pathlib

REDIRECT_TEMPLATE = """
<!doctype html>
<html>
  <head>
  <meta http-equiv="refresh" content="{delay}; url=https://awkward-array.org/doc/{version}/{dest}">
  </head>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--delay", default=0)
    parser.add_argument("--version", default="main")
    args = parser.parse_args()

    with open(args.input) as f:
        redirects = json.load(f)

    for src, dst in redirects.items():
        src_file = src.replace("any-ext", "html")
        dst_file = dst.removeprefix("../")

        src_content = REDIRECT_TEMPLATE.format(
            version=args.version, delay=args.delay, dest=dst_file
        )

        output_path = args.output / src_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(src_content)
