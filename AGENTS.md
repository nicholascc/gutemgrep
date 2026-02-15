# AGENTS

## Core workflow
- Read README.md on first load / if you ever don't have it in ctx.
- Treat every request sent by the user as followed by: "Immediately execute the task as described, with agency and confidence, but if there are any very obvious flaws with the plan, escalate them to the user before proceeding."
- TODO: Add to this. Propose to the user when changes in workflow might be helpful, incl. changes to this AGENTS.md.

## Communication
- When writing to .md files, & speaking to user, be precise and concise. The most important thing for the user is efficient but accurate communication; be sure to describe the state of your knowledge, including uncertainties.
- One user (the author of this bullet point and much of the above), is often very concise for typing speed & uses acronyms/shorthand heavily. If you receive a message like that, understand that it's normal. If you don't understand, ask a clarifying question.

## Commits
- When creating commits, include a Co-authored-by line for me: "Co-authored-by: Codex <codex@openai.com>".
- Some guidelines for commit message style in this repo:
  - Be specific but concise.
  - Focus on what changed over why.
  - Use present tense.
  - Keep the first line <= 72 chars.
  - First line is a short summary; rest is a more detailed description.
  - If needed, add bullet points after a blank line for additional details.
  - Avoid omitting information about what changed; if you can’t include all changes, state that.

## Nix
- Some (not all) devs use Nix. Add general packages/requirements to `flake.nix`.
