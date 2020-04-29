# BSD 3-Clause License; see https://github.com/jpivarski/awkward-1.0/blob/master/LICENSE

import json
import subprocess
import re
import http.client

tagslist_text = subprocess.run(["git", "show-ref", "--tags"], stdout=subprocess.PIPE).stdout
tagslist = dict(re.findall(rb"([0-9a-f]{40}) refs/tags/([0-9\.]+)", tagslist_text))

subjects_text = subprocess.run(["git", "log", "--format='format:%H %s'"], stdout=subprocess.PIPE).stdout
subjects = re.findall(rb"([0-9a-f]{40}) (.*)", subjects_text)

github_connection = http.client.HTTPSConnection("api.github.com")
github_connection.request("GET", "/repos/scikit-hep/awkward-1.0/releases", headers={"User-Agent": "awkward1-changelog"})
github_releases_text = github_connection.getresponse().read()
try:
    github_releases = json.loads(github_releases_text)
except:
    print(github_releases_text)
    raise
releases = {x["tag_name"]: x["body"] for x in github_releases}

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

    numprs = None

    for taghash, subject in subjects:
        if taghash in tagslist:
            tag = tagslist[taghash].decode()
            tagurl = "https://github.com/scikit-hep/awkward-1.0/releases/tag/{0}".format(tag)

            if numprs == 0:
                outfile.write("*(no pull requests)*\n")
            numprs = 0

            if pypi_exists(tag):
                pypi_url = " (`pip <https://pypi.org/project/awkward1/{0}/>`__)".format(tag)
            else:
                pypi_url = ""

            header_text = "\nRelease `{0} <{1}>`__{2}\n".format(tag, tagurl, pypi_url)
            outfile.write(header_text)
            outfile.write("="*len(header_text) + "\n\n")

            if tag in releases:
                text = releases[tag].strip()
                text = re.sub(r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+)#([1-9][0-9]*)", r"`\1#\2 <https://github.com/\1/issues/\2>`__", text)
                text = re.sub(r"([^a-zA-Z0-9\-_])#([1-9][0-9]*)", r"\1`#\2 <https://github.com/scikit-hep/awkward-1.0/issues/\2>`__", text)
                outfile.write(text + "\n\n")

        m = re.match(rb"(.*) \(#([1-9][0-9]*)\)", subject)
        if m is not None:
            if numprs is None:
                numprs = 0
            numprs += 1

            text = m.group(1).decode()
            prnum = m.group(2).decode()
            prurl = "https://github.com/scikit-hep/awkward-1.0/pull/{0}".format(prnum)

            known = []
            for issue in re.findall(r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+)#([1-9][0-9]*)", text):
                known.append(issue)
            for issue in re.findall(r"([^a-zA-Z0-9\-_])#([1-9][0-9]*)", text):
                known.append(issue[1])

            text = re.sub(r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+)#([1-9][0-9]*)", r"`\1#\2 <https://github.com/\1/issues/\2>`__", text)
            text = re.sub(r"([^a-zA-Z0-9\-_])#([1-9][0-9]*)", r"\1`#\2 <https://github.com/scikit-hep/awkward-1.0/issues/\2>`__", text)

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
                addresses_text = " (Addresses {0})".format(", ".join(addresses))

            outfile.write("* PR `#{0} <{1}>`__: {2}{3}\n".format(prnum, prurl, text, addresses_text))
