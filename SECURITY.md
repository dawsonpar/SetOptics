# Security Policy

## Supported versions

This is an actively maintained open-source repository. Security fixes are
applied to `main`. There are no LTS branches.

## Reporting a vulnerability

If you discover a security issue, please **do not file a public GitHub
issue**. Instead, email the maintainer at:

  `dawpar7@gmail.com`

with subject line beginning `[SetOptics security]`. Include:

- A description of the issue and the impact you observed.
- Steps to reproduce, including any required input files.
- The affected commit hash or version, if known.
- Your name and (optional) a link for credit.

You can expect an initial acknowledgement within 7 days. If the issue is
confirmed, a fix will be prepared on a private branch and disclosed
publicly after release, with credit to the reporter if desired.

## Scope

In scope:

- Remote code execution via crafted video or annotation input.
- Path traversal or arbitrary file write in the annotation or detection
  pipelines.
- Prompt injection in the LLM annotation pipeline that causes the agent
  to take unintended actions on the host.
- Leakage of `GEMINI_API_KEY` or other secrets via logs or output files.

Out of scope:

- Vulnerabilities in third-party dependencies. Report those upstream.
- Issues that require the attacker to already have local code execution on
  the host running the pipeline.
- Hardening recommendations without a concrete exploit path. Those are
  welcome as regular GitHub issues.

## No bug bounty

This is a small open-source project with no funded bounty program. Credit
in release notes is the only reward we can offer.
