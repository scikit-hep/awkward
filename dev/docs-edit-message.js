async function update_pr_description(github, context, preview_url) {

    const {data: pull} = await github.rest.pulls.get({
        owner: context.repo.owner,
        repo: context.repo.repo,
        pull_number: context.issue.number,
    });
    const MESSAGE_START = `<!-- docs-preview-start -->`;
    const MESSAGE_END = `<!-- docs-preview-end -->`;
    const MESSAGE_BODY = `----\n:books: The documentation for this PR is available at <${preview_url}/> :sparkles:`;
    const MESSAGE = `\r\n\r\n${MESSAGE_START}\r\n${MESSAGE_BODY}\r\n${MESSAGE_END}`

    // Only include message if this is the first time.
    let body = "";
    if (!pull.body) {
        body = MESSAGE;
    } else if (pull.body.indexOf(MESSAGE_START) === -1) {
        body = pull.body + MESSAGE;
    } else {
        return;
    }
    // Update description
    github.rest.pulls.update({
        owner: context.repo.owner,
        repo: context.repo.repo,
        pull_number: context.issue.number,
        body: body,
    });
}

module.exports = update_pr_description;
