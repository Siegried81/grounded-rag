# Security policy

## Supported versions

This is a portfolio and training project with no releases. Only the latest
commit on `main` is maintained; fixes land there and nowhere else.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Use GitHub's private reporting instead: the **Security** tab of this repository,
then **Report a vulnerability**. If that button is not available, open an issue
titled "Security contact" with no details, and a private channel will be
arranged.

Include what you found, how to reproduce it, and what an attacker gains. You can
expect a first answer within seven days; a confirmed issue is fixed on `main`
and credited in the fix unless you prefer otherwise.

## Secrets

Every key and token is read from a local `.env` that git ignores; `.env.example`
lists the names with empty values. A credential that appears in this
repository's history is a vulnerability in its own right: report it the same
way, and it will be revoked rather than merely deleted.

## What is in scope here

- Every retrieved passage is untrusted input, fenced as data in the prompt;
  forged fence tags inside a passage are escaped.
- In scope: a document or question that makes the model follow injected
  instructions, cite a source that was not retrieved, or reveal its system
  prompt. The red-team suite (`scripts/run_redteam.py`) is the reference for what is tested.
- Keys: `GROQ_API_KEY*`, `OPENROUTER_API_KEY`, `HF_TOKEN`, `HOSTED_EMBED_API_KEY`; none is needed with a local
  Ollama.
