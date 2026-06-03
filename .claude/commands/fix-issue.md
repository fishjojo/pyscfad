---
allowed-tools: Task, Edit, Write, Read, Grep, Glob, Bash(git:*), Bash(gh:*), Bash(./scripts/gh.sh:*), Bash(python:*), Bash(pytest:*), Bash(pip:*)
description: Resolve a GitHub issue (bug fix or feature) via a multi-agent pipeline
---

You are the **orchestrator** for an automated issue-resolution run on the PySCFAD
repository — this handles both **bug fixes** and **feature implementations**. Read
`CLAUDE.md` for build/test/architecture conventions before doing anything else, and
follow it exactly.

This command always runs the full multi-agent pipeline. It is invoked only by a trusted
maintainer (via an `@claude` mention), so assume the work is authorized.

## Inputs

Parse these from the invocation arguments (`$ARGUMENTS`). Use the defaults if a key
is missing.

- `REPO` — `owner/name` of the repository (default: infer from `git remote`).
- `ISSUE_NUMBER` — the issue to resolve (required).
- `GIT` — `self` or `harness` (default: `self`).
  - `self`: **you** create the branch, commit, push, open the PR, and post issue
    comments (you own all GitHub side effects).
  - `harness`: leave all changes in the working tree and do **not** run any git/gh
    **write** command — no commit, push, PR, **or issue comment**. An external harness
    owns every GitHub side effect; it commits/pushes/PRs and reports your result.

**Reporting convention (applies everywhere below, including every early stop).** When a
step says to "report" / "comment" a verdict, summary, or failure:
- `GIT: self` → post it as an issue comment with `gh` (or open the PR, as specified).
- `GIT: harness` → do **not** call `gh`/`git`; return the text as your final message and
  let the harness post it. Reads (`gh issue view`, `./scripts/gh.sh`) are fine in both.

Arguments: `$ARGUMENTS`

## Step 0 — Load the issue

Fetch the issue and its discussion so you understand the actual request:

```
gh issue view <ISSUE_NUMBER> --repo <REPO> --json number,title,body,labels,comments
```

(If `gh` is unavailable but `./scripts/gh.sh` exists, use that wrapper instead.)

Restate, for yourself, what success looks like — for a bug, the expected vs. actual
behavior and the smallest reproduction; for a feature, the desired capability and how it
should be exercised. Bug reports and feature requests are both in scope.

---

## Stage 0 — Verify the bug (single-agent gate)

**Before running the multi-agent pipeline**, dispatch a single `bug-verifier` subagent
with the issue text. It does no fixing; it only decides whether the work is real:

- For a **bug**, it attempts the smallest reproduction and returns
  `confirmed` / `not_reproducible` / `not_a_bug` / `needs_info`.
- For a **feature**, it returns `actionable` / `needs_info` / `out_of_scope`.

**Gate:** Proceed to the multi-agent pipeline **only** if the verdict is `confirmed`
(bug) or `actionable` (feature). For any other verdict, do **not** change code, create a
branch, or open a PR — **report** the verdict and the verifier's evidence (what was
tried, what's missing, or why it isn't a bug) per the **Reporting convention** above
(`GIT: self` → an issue comment; `GIT: harness` → your final message), then stop.

Carry the verifier's reproduction forward as the ground-truth repro for the rest of the
run.

---

## Multi-agent pipeline

You orchestrate four roles. Dispatch each role with the `Task` tool using the matching
subagent type, pass it only what it needs, and integrate the structured result it
returns. Think hard at every integration point.

### Stage 1 — Context (subagent: `context-builder`, medium/high effort)

Dispatch `context-builder` with the issue text, your success criteria, and the confirmed
reproduction from Stage 0. Require it to return: for a bug, the root-cause hypothesis
(building on the known repro); for a feature, where and how it should be implemented. In
both cases: the exact files/symbols involved (`path:line`), existing related tests, and
any architectural constraints from `CLAUDE.md` the change must respect.

### Stage 2 — Plan (subagent: `fix-planner`, highest effort)

Hand the context brief to `fix-planner`. Require a step-by-step plan: an ordered list of
**independent or sequenced slices**, each naming the files to touch, the change, and how
to verify it; plus the overall test strategy and explicit risks/edge cases. Review the
plan critically yourself before proceeding — reject and re-request if a slice is
underspecified.

### Stage 3 — Implement + verify loop (orchestrator, highest effort)

Drive the plan to convergence. For up to **4 iterations**:

1. For each not-yet-done slice, dispatch a `fix-implementer` subagent (medium/high
   effort) with: the slice spec, the relevant `path:line` anchors, and the constraint
   that it edits only files in its slice and reports the diff + reasoning. Run
   independent slices in parallel (multiple `Task` calls in one message); run dependent
   slices in order.
2. After integrating the slices, dispatch a single `fix-verifier` subagent to run the
   targeted tests (and, for a bug, a reproduction of the original failure). It returns
   pass/fail with the failing output.
3. **Converged?** If the verifier passes (and, for a bug, the original repro no longer
   fails), exit the loop. Otherwise, feed the failing output back: decide whether to
   re-dispatch implementers with corrections or revise the plan, and iterate.

If you hit the iteration cap without convergence, do **not** open a green-washed PR.
**Report** (per the **Reporting convention**) what you tried, the remaining failure, and
your best partial diagnosis, then stop. In `GIT: self` you may instead open a **draft**
PR clearly marked "NEEDS WORK" with the failing output; in `GIT: harness`, return that
status as your final message and let the harness decide.

---

## Finalize

Confirm the change is real before claiming success: the new/updated tests pass and, for
a bug, the original reproduction no longer fails. Report failures honestly — never
describe an unverified or failing change as done.

If `GIT: harness`:
- Leave all edits in the working tree. Do **not** run `git commit`/`git push`/`gh pr
  create`. Print a concise summary (what changed, why, how verified) as your final
  message. The harness commits, pushes, and opens the PR.

If `GIT: self`:
1. Create a branch: `agent/fix-<ISSUE_NUMBER>-<short-slug>` off the default branch.
   Never commit to the default branch.
2. Commit with a message referencing the issue, ending with:
   `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
3. Push the branch and open a **draft** PR whose body starts with `Fixes #<ISSUE_NUMBER>`
   and includes: a summary of the change, the plan and per-slice notes, and the exact
   verification commands + their results. End the PR body with:
   `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
4. Post a brief comment on the issue linking the PR.
