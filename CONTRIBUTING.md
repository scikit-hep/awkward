# Contributing to Awkward Array

Thank you for your interest in contributing! We're eager to see your ideas and look forward to working with you.

This document describes the technical procedures we follow in this project. I should also stress that as members of the Scikit-HEP community, we are all obliged to maintaining a welcoming, harassment-free environment. See the [Code of Conduct](https://scikit-hep.org/code-of-conduct) for details.

### Where to start

The front page for the Awkward Array project is its [GitHub README](https://github.com/scikit-hep/awkward-1.0#readme). This leads directly to tutorials and reference documentation that I assume you've already seen. It also includes instructions for [compiling for development](https://github.com/scikit-hep/awkward-1.0#installation-for-developers), using the localbuild.py script.

### Reporting issues

The first thing you should do if you want to fix something is to [submit an issue through GitHub](https://github.com/scikit-hep/awkward-1.0/issues). That way, we can all see it and maybe I or a member of the community knows of a solution that could save you the time spent fixing it. If you want to "own" the issue, you can signal your intent to fix it in the issue report.

### Contributing a pull request

Feel free to [open pull requests in GitHub](https://github.com/scikit-hep/awkward-1.0/pulls) from your forked repo when you start working on the problem. I recommend opening the pull request early so that we can see your progress and communicate about it. (Note that you can `git commit --allow-empty` to make an empty commit and start a pull request before you even have new code.)

Please [make the pull request a draft](https://github.blog/2019-02-14-introducing-draft-pull-requests/) to indicate that it is in an incomplete state and shouldn't be merged until you click "ready for review."

At present, I (Jim Pivarski, [jpivarski](https://github.com/jpivarski)) merge or close all pull requests for Awkward Array, though a team of maintainers should be enlisted in the future, as the project matures. When I'm working closely with a developer, such as a summer student, I'll sometimes give that developer permission to merge their own pull requests.

If you're waiting for me to review, comment upon, or merge a pull request, please do remind me by mentioning me (`@jpivarski`) in a comment. It's possible that I've forgotten and I apologize in advance. (I tend to give the person I'm currently working with my full attention, unfortunately at the expense of others.)

### Becoming a regular committer

If you want to contribute frequently, I'll grant you write access to the `scikit-hep/awkward-1.0` repo itself. This is more convenient than pull requests from forked repos because I can contribute corrections to your branch in fewer steps.

### Git practices

That said, most of the commits on a pull request/git branch should be from a single author. Corrections or suggestions from other authors are exceptional cases, when a particular change is easier to express as a code diff than in words.

As such, you should name your branch starting with your GitHub userid and a slash, such as `jpivarski/write-contributing-md`. If you start a pull request with a branch that doesn't follow convention, though, you don't need to fix it.

### Continuous integration

Pull requests must pass all [continuous integration](https://dev.azure.com/jpivarski/Scikit-HEP/_build?definitionId=3&_a=summary) tests before they are merged. I will sometimes cancel non-essential builds to give priority to pull requests that are almost ready to be merged. If you needed the result of the build as a diagnostic, you can ask me to restart your job or make a trivial change to trigger a new build.

Currently, we only run merge builds (the state of your branch if merged with master), not regular branch builds (the state of your branch as-is), because only merge builds can be made to run for pull requests from external forks and it makes better use of our limited execution time on Azure. If you want to enable regular branch builds, you can turn it on for your branch by editing `trigger/branches/exclude` in [.ci/azure-buildtest-awkwrad.yml](https://github.com/scikit-hep/awkward-1.0/blob/9b6fca3f6e6456860ae40979171f762e0045ce7c/.ci/azure-buildtest-awkward.yml#L1-L5). The merge build trigger is not controlled by the YAML file. It is better, however, to keep up-to-date with `git merge master`.

### Git history

Most pull requests are merged with the "squash and merge" feature, so details about commit history within a pull request are lost. Feel free, therefore, to commit with any frequency you're comfortable with. I like to make frequent commits to avoid losing work to a dead laptop, and to have more save-points to recover from.

It is unnecessary to manually edit (rebase) your commit history. If, however, you do want to save a pull request as multiple commits on `master`, ask me and we'll discuss.

The Awkward Array `master` branch must be kept in an unbroken state. Although the recommended way to install Awkward Array is through pip or conda, the `master` branch on GitHub must always be functional. Pull requests for bug fixes and new features are based on `master`, so it has to work for users to test our proposed changes.

The `master` branch is also never far from the latest released version. The [release history](https://awkward-array.readthedocs.io/en/latest/_auto/changelog.html) shows that each release introduces at most several, sometimes only one, completed pull requests.

Committing directly to `master` is not allowed except for

   * updating the `VERSION_INFO` file, which should be independent of pull requests
   * updating documentation or non-code files
   * unprecedented emergencies

and only by me.

