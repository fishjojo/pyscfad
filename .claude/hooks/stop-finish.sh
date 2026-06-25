#!/usr/bin/env bash
# Stop hook for autonomous @claude CI issue-fix runs.
#
# Two jobs:
#   1. Stop the agent from ending early. Deep into a long headless run it can
#      quit on a status update / "I'll now run X" because it applies interactive
#      turn-taking where no human is present to say "continue".
#   2. Make the AGENT open the pull request itself, so the PR is authored by the
#      Claude GitHub App (claude[bot] / app/claude) — the agent's own gh identity
#      — and NOT by the workflow's fallback step, which can only use the default
#      Actions token (github-actions[bot]).
#
# Mechanism: block the stop (exit 2; stderr is fed back to the model) until a PR
# exists for the current branch, with a hard nudge cap so it can never loop
# forever. The PR check is read-only (works with whatever token gh has); the PR
# *creation* is done by the agent, whose gh authenticates as the Claude App. If
# gh cannot answer, fall back to an explicit completion flag
# (git config --local claude.fixComplete true). After the cap, allow the stop and
# let the workflow backstop capture whatever exists.
set -uo pipefail
cat >/dev/null 2>&1 || true   # drain the hook payload on stdin

MAX_NUDGES=4
COUNTER="${RUNNER_TEMP:-/tmp}/claude_stop_nudges"
n=0
[ -f "$COUNTER" ] && n=$(cat "$COUNTER" 2>/dev/null || echo 0)
case "$n" in ''|*[!0-9]*) n=0 ;; esac

block() {  # $1 = reason fed back to the model
  [ "$n" -ge "$MAX_NUDGES" ] && exit 0       # bounded: give up; the backstop catches it
  printf '%s' "$((n + 1))" > "$COUNTER"
  printf '%s\n' "$1" >&2
  exit 2
}

branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")

# Preferred gate: a PR must exist for this branch, opened by the agent itself
# (gh pr create --draft -> authored by the Claude app). Read-only existence check.
pr_count=""
if [ -n "$branch" ] && [ "$branch" != "main" ] && [ "$branch" != "HEAD" ]; then
  pr_count=$(gh pr list --head "$branch" --state open --json number --jq 'length' 2>/dev/null || echo "")
fi

if [ -n "$pr_count" ] && [ "$pr_count" != "0" ]; then
  exit 0   # you opened the PR -> allow the stop
fi
if [ "$pr_count" = "0" ]; then
  block "Autonomous CI run: you have NOT opened the pull request yet. Open it YOURSELF now: finish any remaining work (final tests/regression, commit, push), then run  gh pr create --draft  so the PR is authored by you (the Claude app). Do NOT just post a 'Create a PR' link, and do NOT leave PR creation to the fallback bot. Then stop."
fi

# gh could not answer (unavailable/unauthenticated) -> fall back to a completion flag.
if [ "$(git config --local --get claude.fixComplete 2>/dev/null || true)" = "true" ]; then
  exit 0
fi
block "Autonomous CI run: do not end your turn on a status update or statement of intent. Finish all remaining work, open the draft PR yourself with  gh pr create --draft  (so it is authored by you, the Claude app), then run  git config --local claude.fixComplete true  and stop."
