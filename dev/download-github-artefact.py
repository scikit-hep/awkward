#!/usr/bin/env python
"""Download and extract a GitHub asset by name and SHA"""
from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import zipfile

import requests


def get_sha_head():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=True
    )
    if result.returncode:
        raise RuntimeError
    return result.stdout.decode().strip()


def iter_artefacts(repo, token, per_page=100):
    headers = {"Authorization": f"token {token}"}
    response = requests.get(
        f"https://api.github.com/repos/{repo}/actions/artifacts",
        params={"per_page": per_page},
        headers=headers,
    )
    response.raise_for_status()
    yield from response.json()["artifacts"]

    # Follow pagination
    while "next" in response.links:
        response = requests.get(
            response.links["next"]["url"],
            headers=headers,
        )
        response.raise_for_status()
        yield from response.json()["artifacts"]


def download_and_extract_artefact(artefact, dest, token):
    response = requests.get(
        artefact["archive_download_url"],
        headers={"Authorization": f"token {token}"},
    )
    response.raise_for_status()

    os.makedirs(dest, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(response.content)) as f:
        f.extractall(path=dest)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="name of org/repo")
    parser.add_argument("artefact", help="regex to match name of artefact")
    parser.add_argument("-s", "--sha", help="SHA of commit. Default to (this) HEAD")
    parser.add_argument("-t", "--token", help="GitHub token with correct scopes")
    parser.add_argument("-d", "--dest", help="path to extract output", default=".")
    args = parser.parse_args(argv)

    if args.token is None:
        token = os.environ["GITHUB_TOKEN"]
    else:
        token = args.token

    if args.sha is None:
        sha = get_sha_head()
    else:
        sha = args.sha

    has_seen_sha = False
    for artefact in iter_artefacts(args.repo, token):
        # If SHA matches
        if artefact["workflow_run"]["head_sha"] == sha:
            has_seen_sha = True

            # If query matches
            if re.match(args.artefact, artefact["name"]):
                break
    else:
        # If we've walked past the SHA in question
        if has_seen_sha:
            raise RuntimeError(
                f"Couldn't find artefact matching {args.artefact!r} for SHA"
            )
        raise RuntimeError(f"Couldn't find SHA matching {sha!r}")

    download_and_extract_artefact(artefact, args.dest, token)


if __name__ == "__main__":
    main()
