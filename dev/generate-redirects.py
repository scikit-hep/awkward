import argparse
import json
import pathlib
import urllib.parse

REDIRECT_TEMPLATE = """
<!doctype html>
<html>
  <head>
  <meta http-equiv="refresh" content="{delay}; url={target}">
  </head>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--base", default="https://awkward-array.org/doc/main/")
    parser.add_argument("--delay", default=0)
    args = parser.parse_args()

    with open(args.input) as f:
        redirects = json.load(f)

    for src, dst in redirects.items():
        src_file = src.replace("any-ext", "html")
        dst_file = dst.removeprefix("../")

        target = urllib.parse.urljoin(args.base, dst_file)

        src_content = REDIRECT_TEMPLATE.format(delay=args.delay, target=target)

        output_path = args.output / src_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(src_content)
