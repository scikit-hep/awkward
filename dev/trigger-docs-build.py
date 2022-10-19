"""
Trigger a build on ReadTheDocs of the given branch
"""
import argparse
import os
import re

import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    args = parser.parse_args()

    version_slug = re.sub(r"[\-/]", "-", args.version)

    url = f"https://readthedocs.org/api/v3/projects/awkward-array/versions/{version_slug}/builds/"
    token = os.environ["RTD_TOKEN"]
    response = requests.post(
        url,
        headers={"Authorization": f"token {token}"},
    )
    print(response)
