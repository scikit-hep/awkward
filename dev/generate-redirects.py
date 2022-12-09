import argparse
import json
import pathlib

REDIRECT_TEMPLATE = """
<!doctype html>
<html>
  <head>
  <meta http-equiv="refresh" content="0; url=https://awkward-array.org/doc/main/{dest}">
  </head>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()

    with open(args.input) as f:
        redirects = json.load(f)

    for src, dst in redirects.items():
        src_file = src.replace("any-ext", "html")
        dst_file = dst.removeprefix("../")

        src_content = REDIRECT_TEMPLATE.format(dest=dst_file)

        output_path = args.output / src_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(src_content)

    (args.output / "index.html").write_text(REDIRECT_TEMPLATE.format(dest=""))
