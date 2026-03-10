"""
cli.py — run_cli with CommandQueue + full debug tracing.

ROOT-CAUSE NOTE
───────────────
Agent.chat() in agents.py strips raw LLM output down to just the text
after ANSWER: before returning it.  That means any COMMAND: lines the LLM
wrote are thrown away before run_cli ever sees them.

The fix applied here: after every active.chat() call we read the last
assistant message directly out of active._history (which holds the raw
LLM output) and scan *that* for COMMAND: lines instead of scanning the
already-stripped answer string.

For Router objects (which don't expose _history directly) we fall back to
scanning the returned answer string — a degraded but still functional path.

DEBUG MODE
──────────
Toggle with /debug.  While on, every step of the pipeline is printed:
  [1] raw LLM text (first 800 chars)
  [2] regex scan result — what matched / why nothing matched
  [3] queue state before execution
  [4] subprocess invocation details + full stdout/stderr
  [5] what gets fed back into agent history
"""
from __future__ import annotations

import re
import subprocess
import collections
import logging
from typing import Deque

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  The regex that finds COMMAND: lines.
#  Intentionally liberal: allows optional leading whitespace.
#  COMMAND must be uppercase — teach the LLM the exact token.
# ─────────────────────────────────────────────────────────────────────────────
_COMMAND_RE = re.compile(r"^\s*COMMAND:\s*(.+)$", re.MULTILINE)


# ═════════════════════════════════════════════════════════════════════════════
#  Debug printer
# ═════════════════════════════════════════════════════════════════════════════

class _Debugger:
    def __init__(self) -> None:
        self.enabled = False
        self._print_fn = print

    def __call__(self, msg: str) -> None:
        if self.enabled:
            self._print_fn(f"[bold blue]⬡ DBG[/] {msg}")

    def header(self, title: str) -> None:
        if self.enabled:
            self._print_fn(f"\n[bold blue]{'─'*10} {title} {'─'*10}[/]")

    def raw_text(self, label: str, text: str, limit: int = 800) -> None:
        if not self.enabled:
            return
        shown = text[:limit]
        if len(text) > limit:
            shown += f"\n… (truncated, total {len(text)} chars)"
        self._print_fn(f"[bold blue]⬡ {label}[/]\n[dim]{shown}[/]")


DBG = _Debugger()


# ═════════════════════════════════════════════════════════════════════════════
#  CommandQueue
# ═════════════════════════════════════════════════════════════════════════════

class CommandQueue:
    def __init__(self) -> None:
        self._queue: Deque[str] = collections.deque()

    # ── Parsing ───────────────────────────────────────────────────────────

    def ingest(self, text: str, source_label: str = "LLM output") -> list[str]:
        """
        Scan *text* for COMMAND: lines, enqueue them, return newly-found list.
        In debug mode prints a line-by-line breakdown explaining every match
        and near-miss.
        """
        DBG.header(f"INGEST  source={source_label}")
        DBG.raw_text("Full text being scanned", text)

        if DBG.enabled:
            DBG("Line-by-line scan:")
            for i, line in enumerate(text.splitlines(), 1):
                m = re.match(r"^\s*COMMAND:\s*(.+)$", line)
                tag = "[green]MATCH [/]" if m else "[dim] skip[/]"
                near = (
                    "command:" in line.lower()
                    and not re.match(r"^\s*COMMAND:", line)
                )
                hint = "  [yellow]← near-miss? wrong case or extra leading text[/]" if near else ""
                DBG(f"  {i:>3}: {tag}  {repr(line)}{hint}")

        found = _COMMAND_RE.findall(text)
        if DBG.enabled:
            if found:
                DBG(f"[green]Regex matched {len(found)} command(s): {found}[/]")
            else:
                DBG(
                    "[red]Regex matched 0 lines.[/]  "
                    "Likely causes:\n"
                    "    • LLM wrote 'command:' (lowercase)\n"
                    "    • LLM wrapped it in backticks: `COMMAND: ls`\n"
                    "    • COMMAND: is inside prose, not on its own line\n"
                    "    • Agent.chat() stripped it before we got here\n"
                    "  → Run /debug-test to confirm the regex works, then check "
                    "the raw LLM output shown above."
                )

        for cmd in found:
            cmd = cmd.strip()
            if cmd:
                self._queue.append(cmd)
        return found

    # ── Execution ─────────────────────────────────────────────────────────

    def execute_all(
        self,
        *,
        cwd: str | None = None,
        timeout: int = 30,
        print_fn=None,
    ) -> list[dict]:
        DBG.header(f"EXECUTE QUEUE  depth={len(self._queue)}")
        if not self._queue:
            DBG("[yellow]Queue is empty — nothing to execute.[/]")
            return []

        results = []
        while self._queue:
            cmd = self._queue.popleft()
            DBG(f"Dequeuing: [cyan]{cmd!r}[/]  (remaining after pop: {len(self._queue)})")
            if print_fn:
                print_fn(f"[dim]▶ {cmd}[/]")
            result = _run_direct(cmd, cwd=cwd, timeout=timeout)
            results.append(result)
            if print_fn:
                _pretty_result(result, print_fn)
        return results

    # ── Inspection ────────────────────────────────────────────────────────

    def pending(self) -> list[str]:
        return list(self._queue)

    def clear(self) -> None:
        self._queue.clear()

    def __len__(self) -> int:
        return len(self._queue)


# ═════════════════════════════════════════════════════════════════════════════
#  Direct subprocess execution (inside container — no docker exec needed)
# ═════════════════════════════════════════════════════════════════════════════

def _run_direct(command: str, *, cwd: str | None = None, timeout: int = 30) -> dict:
    DBG.header("SUBPROCESS")
    DBG(f"command : [cyan]{command!r}[/]")
    DBG(f"cwd     : {cwd!r}")
    DBG(f"timeout : {timeout}s  |  shell=True")

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
        result = {
            "command":    command,
            "returncode": proc.returncode,
            "stdout":     proc.stdout.strip(),
            "stderr":     proc.stderr.strip(),
            "ok":         proc.returncode == 0,
        }
        DBG(f"rc={proc.returncode}  stdout={result['stdout']!r}  stderr={result['stderr']!r}")
        return result

    except subprocess.TimeoutExpired:
        DBG(f"[red]TIMED OUT after {timeout}s[/]")
        return {"command": command, "returncode": -1,
                "stdout": "", "stderr": f"timed out after {timeout}s", "ok": False}
    except Exception as exc:
        DBG(f"[red]EXCEPTION: {exc}[/]")
        return {"command": command, "returncode": -1,
                "stdout": "", "stderr": str(exc), "ok": False}


def _pretty_result(result: dict, print_fn) -> None:
    ok     = result["ok"]
    colour = "green" if ok else "red"
    badge  = "✓" if ok else "✗"
    rc     = result["returncode"]
    if result["stdout"]:
        print_fn(f"[{colour}]{badge} stdout (rc={rc}):[/{colour}] {result['stdout']}")
    if result["stderr"]:
        print_fn(f"[red]  stderr: {result['stderr']}[/]")
    if not result["stdout"] and not result["stderr"]:
        print_fn(f"[{colour}]{badge} (no output, rc={rc})[/{colour}]")


# ═════════════════════════════════════════════════════════════════════════════
#  Raw-output recovery
#
#  THE CORE FIX:
#  Agent.chat() in agents.py does:
#      m = _ANSWER_RE.search(raw)          # finds "ANSWER: ..."
#      return m.group(1).strip()           # returns ONLY the answer text
#
#  Any COMMAND: lines the LLM emitted are silently discarded at this point.
#
#  We recover them by reading the last assistant entry in active._history,
#  which agents.py appends *before* extracting the answer.
# ═════════════════════════════════════════════════════════════════════════════

def _get_raw_llm_output(active, fallback: str) -> tuple[str, str]:
    """Return (raw_text, source_label).

    For Agent objects: read active._history directly.
    For Router objects: read _last_agent._history (the agent that actually ran).
    """
    # If active is a Router, use the last delegated agent's history instead
    source_obj = active
    source_label_prefix = "active"
    last_agent = getattr(active, "_last_agent", None)
    if last_agent is not None:
        source_obj = last_agent
        source_label_prefix = f"router._last_agent ({last_agent.name})"
        DBG(f"[cyan]Router path: reading history from last agent: {last_agent.name}[/]")

    history = getattr(source_obj, "_history", None)
    if history:
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                raw = msg.get("content", "")
                DBG(
                    f"[green]Raw text source: {source_label_prefix}._history  "
                    f"(last assistant msg, {len(raw)} chars)[/]"
                )
                return raw, f"{source_label_prefix}._history (raw LLM output)"

    DBG(
        "[yellow]_history not accessible on this active object.  "
        "Falling back to stripped answer string.  COMMAND: lines may have been "
        "removed already by Agent.chat().[/]"
    )
    return fallback, "stripped answer (fallback)"


# ═════════════════════════════════════════════════════════════════════════════
#  System-prompt injection
# ═════════════════════════════════════════════════════════════════════════════

_COMMAND_QUEUE_PROMPT = """
COMMAND QUEUE PROTOCOL
======================
To run a shell command emit a line in EXACTLY this format on its OWN line
(no indentation, no bullet, no backtick, no prose around it):

COMMAND: <the full shell command>

Multiple COMMAND: lines are executed in order immediately after your reply.
COMMAND must be UPPERCASE. Place COMMAND: lines BEFORE the ANSWER: line.

SHELL RULES — READ CAREFULLY:
------------------------------
1. Write multi-line files with printf, NOT echo.
   echo does NOT expand \\n into newlines.
   CORRECT:   COMMAND: printf 'line1\\nline2\\n' > file.py
   WRONG:     COMMAND: echo "line1\\nline2" > file.py   <- writes literal \\n

2. To run a Python file:   COMMAND: python3 path/to/file.py
   NEVER use pip to run a file. pip is only for installing packages.

3. Do NOT invent commands that are not real shell programs.
   Available: echo, printf, cat, ls, mkdir, mv, cp, rm, python3, bash, grep, find, chmod

4. If a command fails, do NOT repeat it unchanged. Fix the error first.

5. Never use write_file — it does not exist. Use printf or cat with a heredoc.

EXAMPLES:

Write and run a Python file:
COMMAND: printf 'class Dog:\\n    def bark(self):\\n        print("Woof!")\\n\\nDog().bark()\\n' > dog.py
COMMAND: python3 dog.py
ANSWER: Created dog.py and ran it.

List files:
COMMAND: ls -la
ANSWER: Listed the workspace.
"""


def inject_command_queue_prompt(system_prompt: str) -> str:
    return system_prompt.rstrip() + "\n\n" + _COMMAND_QUEUE_PROMPT.strip() + "\n"


# ═════════════════════════════════════════════════════════════════════════════
#  run_cli
# ═════════════════════════════════════════════════════════════════════════════

def run_cli(cfg: dict, agent_name: str) -> None:
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    from agents import build

    console = Console()
    DBG._print_fn = console.print

    cq = CommandQueue()

    console.print("\n[bold green]🌿 Nausicaa[/]  —  Ctrl+C to quit")
    console.print(
        "[dim]Commands: /reset  /memory  /agents  /tree  "
        "/queue  /run-queue  /debug  /debug-test  exit[/]\n"
    )

    agents, router = build(cfg)

    # Resolve workspace so the LLM always knows the exact path and
    # can't hallucinate typos like "coney" instead of "conway".
    import os
    from pathlib import Path
    workspace_path = str(Path(cfg.get("workspace", ".")).expanduser().resolve())
    os.chdir(workspace_path)   # make cwd == workspace so relative paths work too

    for agent in agents.values():
        agent._system = inject_command_queue_prompt(agent._system)
        # Stamp the exact workspace path into every agent's system prompt
        agent._system = (
            f"WORKSPACE DIRECTORY (use this exact path, do not guess): {workspace_path}\n\n"
            + agent._system
        )

    if agent_name == "router" and router:
        active = router
        console.print(f"[dim]Mode: router → {list(agents.keys())}[/]\n")
    elif agent_name in agents:
        active = agents[agent_name]
        console.print(f"[dim]Agent: {agent_name}[/]\n")
    else:
        active = agents.get("default") or next(iter(agents.values()))
        console.print(f"[dim]Agent: default[/]\n")

    while True:
        try:
            user = Prompt.ask("[bold cyan]you[/]")
        except (KeyboardInterrupt, EOFError):
            active.reset()
            console.print("\n[dim]Session saved. bye[/]")
            break

        u = user.strip()
        if not u:
            continue

        if u.lower() in ("exit", "quit", "q"):
            active.reset()
            break

        # ── /debug — toggle verbose pipeline tracing ───────────────────────
        if u == "/debug":
            DBG.enabled = not DBG.enabled
            state = "[green]ON[/]" if DBG.enabled else "[red]OFF[/]"
            console.print(f"[dim]Debug mode: {state}[/]")
            if DBG.enabled:
                console.print(
                    "[dim]You will now see:\n"
                    "  • Raw LLM output (pre-ANSWER-strip)\n"
                    "  • Line-by-line regex scan with near-miss hints\n"
                    "  • Queue depth before/after each turn\n"
                    "  • Exact subprocess call + rc/stdout/stderr\n"
                    "  • What gets injected back into agent history[/]"
                )
            continue

        # ── /debug-test — sanity-check regex + subprocess independently ────
        if u == "/debug-test":
            console.print("[bold blue]── Debug self-test ──[/]")
            test_text = "COMMAND: echo hello_from_debug_test\nANSWER: testing"
            console.print(f"[dim]Synthetic input: {test_text!r}[/]")
            old_dbg = DBG.enabled
            DBG.enabled = True            # force debug for the self-test
            found = cq.ingest(test_text, source_label="debug-test input")
            if found:
                console.print(f"[green]✓ Regex matched: {found}[/]")
                results = cq.execute_all(print_fn=console.print)
                if results and results[0]["ok"]:
                    console.print(
                        f"[green]✓ Subprocess OK  stdout={results[0]['stdout']!r}[/]"
                    )
                else:
                    console.print(f"[red]✗ Subprocess failed: {results}[/]")
            else:
                console.print(
                    "[red]✗ Regex matched nothing — the pattern is broken.[/]"
                )
            DBG.enabled = old_dbg
            continue

        if u == "/reset":
            active.reset()
            cq.clear()
            console.print("[dim]Session saved, queue cleared.[/]")
            continue

        if u == "/agents":
            console.print(f"[dim]Available: {list(agents.keys())}[/]")
            continue

        if u.startswith("/agent "):
            name = u.split(None, 1)[1].strip()
            if name in agents:
                active = agents[name]
                console.print(f"[dim]Switched to: {name}[/]")
            else:
                console.print(f"[dim]Unknown. Available: {list(agents.keys())}[/]")
            continue

        if u == "/memory":
            a = next(iter(agents.values()))
            txt = a._memory.facts.read_text() if a._memory.facts.exists() else "(empty)"
            console.print(f"[dim]{txt}[/]")
            continue

        if u == "/tree":
            console.print(
                "[dim]" + next(iter(agents.values()))._memory.root.parent.name + "[/]"
            )
            continue

        if u == "/queue":
            pending = cq.pending()
            if not pending:
                console.print("[dim](queue is empty)[/]")
            else:
                console.print(f"[dim]Pending ({len(pending)}):[/]")
                for i, cmd in enumerate(pending, 1):
                    console.print(f"[dim]  {i}. {cmd}[/]")
            continue

        if u == "/run-queue":
            if not cq:
                console.print("[dim]Queue is empty.[/]")
            else:
                cq.execute_all(cwd=workspace_path, print_fn=console.print)
            continue

        # ── Chat ──────────────────────────────────────────────────────────
        try:
            DBG.header("CHAT TURN START")
            MAX_RETRY = 6   # max self-correction attempts per user turn

            for _attempt in range(MAX_RETRY):
                DBG(f"[cyan]Attempt {_attempt + 1}/{MAX_RETRY}[/]")

                # ── Pre-flight: strip stale COMMAND_QUEUE_RESULTS ─────────
                _hist_obj = getattr(active, "_last_agent", None) or active
                if hasattr(_hist_obj, "_history"):
                    before  = len(_hist_obj._history)
                    cleaned = []
                    for msg in _hist_obj._history:
                        role    = msg.get("role", "")
                        content = str(msg.get("content", ""))
                        if role == "user" and content.startswith("COMMAND_QUEUE_RESULTS"):
                            continue
                        if role == "assistant" and "\n\nCOMMAND_QUEUE_RESULTS:" in content:
                            msg = dict(msg)
                            msg["content"] = content.split("\n\nCOMMAND_QUEUE_RESULTS:")[0]
                        cleaned.append(msg)
                    _hist_obj._history = cleaned
                    pruned = before - len(_hist_obj._history)
                    DBG(f"[yellow]Pre-flight prune: {before} -> {len(_hist_obj._history)} "
                        f"({pruned} dropped)[/]")

                    if len(_hist_obj._history) > 8:
                        _hist_obj._history = _hist_obj._history[-8:]
                        DBG("[yellow]Trimmed history to last 8 messages[/]")

                # ── On retry turns, inject the error as a new user message ─
                # so the LLM knows what went wrong and can self-correct.
                if _attempt > 0 and _failed_results:
                    error_summary = "\n".join(
                        f"Command `{r['command']}` failed (rc={r['returncode']}): "
                        f"{r['stderr'] or r['stdout'] or 'no output'}"
                        for r in _failed_results
                    )
                    retry_msg = (
                        f"The previous command(s) failed:\n{error_summary}\n"
                        f"Fix the error and try again. "
                        f"If a package is missing, install it first with the correct installer "
                        f"(apt-get, pip, etc.), then retry the original command."
                    )
                    console.print(f"[yellow]↺ retry {_attempt}: {error_summary[:120]}[/]")
                    DBG(f"Injecting retry context: {retry_msg[:200]}")
                    answer = active.chat(retry_msg)
                else:
                    answer = active.chat(u)

                DBG(f"active.chat() returned {len(answer)} chars: {answer[:300]!r}")

                # ── Recover raw LLM output ─────────────────────────────────
                raw_llm, raw_source = _get_raw_llm_output(active, fallback=answer)

                # ── Ingest COMMAND: lines ──────────────────────────────────
                new_cmds = cq.ingest(raw_llm, source_label=raw_source)
                if new_cmds:
                    console.print(
                        f"[dim]Queued {len(new_cmds)}: "
                        + "  ".join(f"[cyan]{c}[/cyan]" for c in new_cmds)
                        + "[/]"
                    )
                else:
                    DBG("[yellow]No COMMAND: lines found this turn.[/]")

                # ── Execute queue ──────────────────────────────────────────
                _failed_results = []
                if cq:
                    console.print(f"\n[bold magenta]⚙  executing {len(cq)} command(s)…[/]")
                    results = cq.execute_all(cwd=workspace_path, print_fn=console.print)
                    _failed_results = [r for r in results if not r["ok"]]

                    # Attach results to the last assistant message
                    _hist_target = (
                        getattr(active, "_last_agent", None) or active
                        if not hasattr(active, "_history")
                        else active
                    )
                    if hasattr(_hist_target, "_history") and _hist_target._history:
                        active_hist = _hist_target
                    else:
                        active_hist = active

                    if hasattr(active_hist, "_history") and active_hist._history:
                        lines = [
                            f"COMMAND `{r['command']}` rc={r['returncode']}\n"
                            f"stdout: {r['stdout'] or '(none)'}\n"
                            f"stderr: {r['stderr'] or '(none)'}"
                            for r in results
                        ]
                        results_block = "\n\nCOMMAND_QUEUE_RESULTS:\n" + "\n\n".join(lines)
                        for i in range(len(active_hist._history) - 1, -1, -1):
                            if active_hist._history[i].get("role") == "assistant":
                                active_hist._history[i]["content"] += results_block
                                DBG(f"Appended results to history[{i}]")
                                break
                        else:
                            active_hist._history.append(
                                {"role": "user", "content": "COMMAND_QUEUE_RESULTS:\n" + "\n\n".join(lines)}
                            )
                else:
                    _failed_results = []
                    DBG("Queue empty — skipping execution.")

                # ── Done when no failures AND answer looks final ────────
                _answer_has_artifacts = bool(
                    re.search(r"^(TOOL_RESULT|TOOL:|ARGS:)", answer, re.MULTILINE)
                )
                if not _failed_results and not _answer_has_artifacts:
                    # Commands ran successfully. Ask the LLM to summarise using
                    # the REAL stdout — this replaces the pre-execution prediction.
                    if results:
                        real_output = "\n".join(
                            f"Command: {r['command']}\n"
                            f"stdout: {r['stdout'] or '(no output)'}\n"
                            f"stderr: {r['stderr'] or '(none)'}"
                            for r in results
                        )
                        summary_prompt = (
                            f"The commands finished. Here is the REAL output:\n"
                            f"{real_output}\n"
                            f"Summarise what happened using only the output above. "
                            f"Do not invent or predict any output."
                        )
                        DBG(f"[green]Fetching grounded summary from LLM[/]")
                        answer = active.chat(summary_prompt)
                        DBG(f"Grounded answer: {answer[:200]!r}")
                        # Remove the summary_prompt exchange from history so it
                        # doesn't replay on the next user turn.
                        # active.chat() appended: user(summary_prompt) + assistant(answer)
                        _hist_obj2 = getattr(active, "_last_agent", None) or active
                        if hasattr(_hist_obj2, "_history") and len(_hist_obj2._history) >= 2:
                            _hist_obj2._history = _hist_obj2._history[:-2]
                            DBG("[yellow]Popped summary exchange from history[/]")
                    DBG(f"[green]All commands succeeded on attempt {_attempt + 1}[/]")
                    break
                if _answer_has_artifacts and not _failed_results:
                    DBG("[yellow]Answer contains tool artifacts — continuing[/]")
                    _failed_results = [{"command": "(pending)", "returncode": 1,
                                        "stdout": "", "stderr": "task not yet complete"}]

                DBG(f"[red]{len(_failed_results)} command(s) failed — will retry[/]")

            else:
                console.print(
                    f"[bold red]⚠  gave up after {MAX_RETRY} attempts. "
                    f"Last errors: "
                    + ", ".join(r['stderr'][:80] for r in _failed_results)
                    + "[/]"
                )

            # ── Display final answer ───────────────────────────────────────
            # Strip COMMAND: lines and TOOL_RESULT lines from display
            display = _COMMAND_RE.sub("", answer)
            display = re.sub(r"^TOOL_RESULT\s*\([^)]*\):.*$", "", display, flags=re.MULTILINE)
            display = re.sub(r"^TOOL:\s*\S+.*$", "", display, flags=re.MULTILINE)
            display = re.sub(r"^ARGS:\s*\{.*$", "", display, flags=re.MULTILINE)
            display = display.strip()
            if display:
                console.print("\n[bold yellow]nausicaa:[/]")
                console.print(Markdown(display))
                console.print()
            else:
                DBG("Answer empty after stripping COMMAND: lines.")

        except Exception as e:
            import traceback
            console.print(f"[bold red]error:[/] {e}\n{traceback.format_exc()}")
