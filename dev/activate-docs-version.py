import argparse
import os
import re

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("-s", "--show", action="store_true")
    args = parser.parse_args()

    version_slug = re.sub(r"[\-/]", "-", args.version)

    url = f"https://readthedocs.org/api/v3/projects/awkward-array/versions/{version_slug}/"
    token = os.environ["RTD_TOKEN"]
    response = requests.patch(
        url,
        json={"active": True, "hidden": not args.show},
        headers={"Authorization": f"token {token}"},
    )
    response.raise_for_status()
