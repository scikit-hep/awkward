# BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/master/LICENSE

import math
import json
import subprocess
import re
import http.client
import datetime

tagslist_text = subprocess.run(["git", "show-ref", "--tags"], stdout=subprocess.PIPE).stdout
tagslist = dict(re.findall(rb"([0-9a-f]{40}) refs/tags/([0-9\.]+)", tagslist_text))

subjects_text = subprocess.run(["git", "log", "--format='format:%H %s'"], stdout=subprocess.PIPE).stdout
subjects = re.findall(rb"([0-9a-f]{40}) (.*)", subjects_text)

github_connection = http.client.HTTPSConnection("api.github.com")
github_releases = []
numpages = int(math.ceil(len(tagslist) / 30.0))
for pageid in range(numpages):
    print("Requesting GitHub data, page {0} of {1}".format(pageid + 1, numpages))
    github_connection.request("GET", r"/repos/scikit-hep/awkward-1.0/releases?page={0}&per_page=30".format(pageid + 1), headers={"User-Agent": "awkward1-changelog"})
    github_releases_text = github_connection.getresponse().read()
    try:
        github_releases_page = json.loads(github_releases_text)
    except:
        print(github_releases_text)
        raise
    print(len(github_releases_page))
    github_releases.extend(github_releases_page)

releases = {x["tag_name"]: x["body"] for x in github_releases if "tag_name" in x and "body" in x}
dates = {x["tag_name"]: datetime.datetime.fromisoformat(x["published_at"].rstrip("Z")).strftime("%A, %d %B, %Y") for x in github_releases if "tag_name" in x and "published_at" in x}
tarballs = {x["tag_name"]: x["tarball_url"] for x in github_releases if "tag_name" in x and "tarball_url" in x}
zipballs = {x["tag_name"]: x["zipball_url"] for x in github_releases if "tag_name" in x and "zipball_url" in x}

pypi_connection = http.client.HTTPSConnection("pypi.org")

def pypi_exists(tag):
    print("Looking for release {0} on PyPI...".format(tag))
    pypi_connection.request("HEAD", "/project/awkward1/{0}/".format(tag))
    response = pypi_connection.getresponse()
    response.read()
    return response.status == 200

with open("_auto/changelog.rst", "w") as outfile:
    outfile.write("Release history\n")
    outfile.write("---------------\n")

    first = True
    numprs = None

    for taghash, subject in subjects:
        if taghash in tagslist:
            tag = tagslist[taghash].decode()
            tagurl = "https://github.com/scikit-hep/awkward-1.0/releases/tag/{0}".format(tag)

            if numprs == 0:
                outfile.write("*(no pull requests)*\n")
            numprs = 0

            header_text = "\nRelease `{0} <{1}>`__\n".format(tag, tagurl)
            outfile.write(header_text)
            outfile.write("="*len(header_text) + "\n\n")

            if tag in dates:
                date_text = "**" + dates[tag] + "**"
            else:
                date_text = ""

            assets = []
            if pypi_exists(tag):
                assets.append("`pip <https://pypi.org/project/awkward1/{0}/>`__".format(tag))
            if tag in tarballs:
                assets.append("`tar <{0}>`__".format(tarballs[tag]))
            if tag in zipballs:
                assets.append("`zip <{0}>`__".format(zipballs[tag]))
            if len(assets) == 0:
                assets_text = ""
            else:
                assets_text = " ({0})".format(", ".join(assets))

            if len(date_text) + len(assets_text) > 0:
                outfile.write("{0}{1}\n\n".format(date_text, assets_text))

            if tag in releases:
                text = releases[tag].strip().replace("For details, see the [release history](https://awkward-array.readthedocs.io/en/latest/_auto/changelog.html).", "")
                text = re.sub(r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+)#([1-9][0-9]*)", r"`\1#\2 <https://github.com/\1/issues/\2>`__", text)
                text = re.sub(r"([^a-zA-Z0-9\-_])#([1-9][0-9]*)", r"\1`#\2 <https://github.com/scikit-hep/awkward-1.0/issues/\2>`__", text)
                outfile.write(text + "\n\n")

            first = False

        m = re.match(rb"(.*) \(#([1-9][0-9]*)\)", subject)
        if m is not None:
            if numprs is None:
                numprs = 0
            numprs += 1

            if first:
                header_text = "\nUnreleased (`master branch <https://github.com/scikit-hep/awkward-1.0>`__ on GitHub)\n"
                outfile.write(header_text)
                outfile.write("="*len(header_text) + "\n\n")

            text = m.group(1).decode().strip()
            prnum = m.group(2).decode()
            prurl = "https://github.com/scikit-hep/awkward-1.0/pull/{0}".format(prnum)

            known = [prnum]
            for issue in re.findall(r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+)#([1-9][0-9]*)", text):
                known.append(issue)
            for issue in re.findall(r"([^a-zA-Z0-9\-_])#([1-9][0-9]*)", text):
                known.append(issue[1])

            text = re.sub(r"`", "``", text)
            text = re.sub(r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+)#([1-9][0-9]*)", r"`\1#\2 <https://github.com/\1/issues/\2>`__", text)
            text = re.sub(r"([^a-zA-Z0-9\-_])#([1-9][0-9]*)", r"\1`#\2 <https://github.com/scikit-hep/awkward-1.0/issues/\2>`__", text)
            if re.match(r".*[a-zA-Z0-9_]$", text):
                text = text + "."

            body_text = subprocess.run(["git", "log", "-1", taghash.decode(), "--format='format:%b'"], stdout=subprocess.PIPE).stdout.decode()
            addresses = []
            for issue in re.findall(r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+)#([1-9][0-9]*)", body_text):
                if issue not in known:
                    addresses.append("`{0}#{1} <https://github.com/{0}/issues/{1}>`__".format(issue[0], issue[1]))
            for issue in re.findall(r"([^a-zA-Z0-9\-_])#([1-9][0-9]*)", body_text):
                if issue[1] not in known:
                    addresses.append("`#{0} <https://github.com/scikit-hep/awkward-1.0/issues/{0}>`__".format(issue[1]))
            if len(addresses) == 0:
                addresses_text = ""
            else:
                addresses_text = " (**also:** {0})".format(", ".join(addresses))

            outfile.write("* PR `#{0} <{1}>`__: {2}{3}\n".format(prnum, prurl, text, addresses_text))

            first = False
