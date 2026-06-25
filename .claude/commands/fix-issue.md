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
(bug) or `actionable` (feature). For any other verdict (`not_reproducible`, `not_a_bug`,
`needs_info`, `out_of_scope`), do **not** change code, create a branch, or open a PR.
The verifier is read-only and does not comment — **you** surface its result: post the
verifier's comment-ready summary per the **Reporting convention** above (`GIT: self` → an
issue comment; `GIT: harness` → your final message), then stop. This is the path that
tells the reporter the bug couldn't be verified / the feature can't be implemented as
specified, so don't drop it.

Carry the verifier's reproduction forward as the ground-truth repro for the rest of the
run.

---

## Stage 0.5 — Open the draft PR first (`GIT: self` only)

**Push first so a partial run never vanishes.** Open the branch + draft PR *now*, as your
first action after the gate — before context-building, planning, or any code. The recurring
failure this prevents: a long run that finishes the work but ends before it ever pushes,
leaving a green job with no branch and no PR. (`GIT: harness`: **skip this stage entirely** —
make no `git`/`gh` writes; the harness owns every side effect.)

For `GIT: self`:
1. **Use the branch you are on; don't fight a pre-created one.** If a CI/harness environment
   has already checked you out onto a non-default branch (e.g. `claude/issue-<N>-<timestamp>`),
   **use that branch** — do **not** create a new one (`git rev-parse --abbrev-ref HEAD` reads
   its name). Only if you are on the default branch, create
   `agent/fix-<ISSUE_NUMBER>-<short-slug>` (append a unique suffix if a remote ref by that name
   already exists). Never commit to the default branch.
2. **Dedup.** If this issue already has an open bot PR
   (`gh pr list --state open --search "Fixes #<ISSUE_NUMBER> in:body"`), reuse it rather than
   opening a second one.
3. **Identity + empty initial commit.** If `git config user.email` is unset, set a local
   identity (`git config user.email "claude[bot]@users.noreply.github.com"` and
   `git config user.name "claude[bot]"`). Make an **empty** initial commit so the branch can
   carry a PR before any code lands:
   `git commit --allow-empty -m "WIP: start fix for #<ISSUE_NUMBER>"`. End every commit
   message with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
4. **Push, then actually create the draft PR.** Push the current branch (prefer the
   environment's push helper if one is provided — the CI workflow names it — otherwise
   `git push -u origin HEAD`). Then **create** the PR with `gh pr create --draft` (do *not*
   merely post a "Create a PR" link): body begins with `Fixes #<ISSUE_NUMBER>`, marked
   **🚧 Work in progress — do not merge**, ending with
   `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. **Record the PR number**
   — you backfill the plan into it after Stage 2 and finalize it later; never open a second PR.

**Workflow-file caveat:** your token cannot push changes under `.github/workflows/`. If a
slice must edit a workflow file, keep that hunk out of the branch and post it as a diff in a
comment for a human to apply — the rest of the change still goes through this flow.

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

(`GIT: self`) Backfill this plan into the draft PR you opened in Stage 0.5: update its body
with the ordered slices and test strategy via `gh pr edit`.

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
3. **Preserve progress.** For `GIT: self`, commit the integrated slices and push them to
   the Stage 0.5 branch **every iteration**, so the remote and the draft PR always reflect
   the latest work even if the run ends mid-loop. For `GIT: harness`, skip this and leave
   the changes in the working tree.
4. **Converged?** If the verifier passes (and, for a bug, the original repro no longer
   fails), exit the loop. Otherwise, feed the failing output back: decide whether to
   re-dispatch implementers with corrections or revise the plan, and iterate.

If you hit the iteration cap without convergence, do **not** green-wash. **Report** (per
the **Reporting convention**) what you tried, the remaining failure, and your best partial
diagnosis. In `GIT: self`, push what you have and leave the **existing** draft PR (from
Stage 0.5) clearly marked "NEEDS WORK" with the failing output, then go to Finalize. In
`GIT: harness`, return that status as your final message and let the harness decide.

---

## Finalize

Confirm the change is real before claiming success: the new/updated tests pass and, for
a bug, the original reproduction no longer fails. Report failures honestly — never
describe an unverified or failing change as done.

If `GIT: harness`:
- Leave all edits in the working tree. Do **not** run `git commit`/`git push`/`gh pr
  create`. Print a concise summary (what changed, why, how verified) as your final
  message. The harness commits, pushes, and opens the PR.

If `GIT: self` (the branch and draft PR already exist from Stage 0.5 — **update** them,
never open a second PR):
1. Commit and push any remaining uncommitted changes to that branch, ending the commit
   message with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
2. Update the **same** draft PR with `gh pr edit <number>`: replace the body with a
   summary of the change, the plan and per-slice notes, and the exact verification
   commands + their results. Keep `Fixes #<ISSUE_NUMBER>` at the top and end the body with:
   `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
3. Set the WIP marker to reflect the real outcome:
   - **Converged:** remove the 🚧 marker and note "verified — ready for review" (leave it
     a draft for a human to mark ready/merge).
   - **Not converged** (hit the iteration cap): mark it **NEEDS WORK** with the failing
     output. Do not green-wash.
4. Post a brief comment on the issue linking the PR.
