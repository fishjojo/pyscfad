#!/usr/bin/env bash
# Stop hook for autonomous @claude CI issue-fix runs.
#
# Problem it solves: deep into a long headless run the model can end its turn on
# a status update or a statement of intent ("I'll now run the regression")
# instead of doing the work — it applies interactive turn-taking where no human
# is present to say "continue". This hook intercepts the stop and pushes the
# agent to finish, with a hard cap so it can never loop forever.
#
# Behaviour:
#   - Allow the stop once the agent has explicitly signalled completion via
#       git config --local claude.fixComplete true
#   - Otherwise block (exit 2; stderr is fed back to the model) up to MAX_NUDGES
#     times, then allow — the workflow backstop captures whatever work exists.
set -uo pipefail
cat >/dev/null 2>&1 || true   # drain the hook payload on stdin

# Completion signalled -> let it stop.
if [ "$(git config --local --get claude.fixComplete 2>/dev/null || true)" = "true" ]; then
  exit 0
fi

MAX_NUDGES=3
COUNTER="${RUNNER_TEMP:-/tmp}/claude_stop_nudges"
n=0
[ -f "$COUNTER" ] && n=$(cat "$COUNTER" 2>/dev/null || echo 0)
case "$n" in ''|*[!0-9]*) n=0 ;; esac
if [ "$n" -ge "$MAX_NUDGES" ]; then
  exit 0   # give up gracefully; do not loop forever
fi
printf '%s' "$((n + 1))" > "$COUNTER"

# Block this stop and tell the model what to finish.
echo "Autonomous CI run: no human will reply, so do NOT end your turn on a status update, a plan, or a stated intention. If any step is unfinished - the final tests/regression, committing, pushing, or opening/updating the PR - do it now with tool calls. When the whole task is genuinely complete and verified, run: git config --local claude.fixComplete true   and then stop. Do not ask permission for reversible steps." >&2
exit 2
