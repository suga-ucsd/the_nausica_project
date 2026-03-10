"""
agents.py — Docker sandbox, tools, Agent, multi-agent Router.

One Agent = one system prompt + one LLM server + shared tools + own memory.
Router = an Agent whose only job is to pick which other Agent handles a request.

Adding a new agent:
  1. Add entry under `agents:` in config.yaml (name, server, system prompt)
  2. Optionally add a dedicated llama-server entry under `servers:`
  3. Router picks it automatically — no code changes needed.
"""
from __future__ import annotations

import atexit
import json
import logging
import re
import shlex
import shutil
import subprocess
import uuid
from datetime import date
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

_CONTAINER_WS = "/workspace"

# ══════════════════════════════════════════════════════════════════════════════
#  Docker sandbox — one shared instance across all agents
# ══════════════════════════════════════════════════════════════════════════════

class Sandbox:
    def __init__(self, workspace: Path, cfg: dict) -> None:
        self.root = workspace.expanduser().resolve()
        self._cfg  = cfg
        self._name = f"nausicaa_{uuid.uuid4().hex[:8]}"
        if not shutil.which("docker"):
            raise RuntimeError("docker not found — install it and add yourself to the docker group")

    def start(self) -> None:
        img = self._cfg.get("image", "debian:bookworm-slim")
        subprocess.run(["docker", "pull", "-q", img], timeout=120, check=False)
        r = subprocess.run([
            "docker", "run", "--detach",
            "--name", self._name,
            "--network",     self._cfg.get("network", "none"),
            "--memory",      self._cfg.get("memory",  "512m"),
            "--cpus",        str(self._cfg.get("cpus", "1.0")),
            "--pids-limit",  str(self._cfg.get("pids", 128)),
            "--security-opt","no-new-privileges:true",
            "--read-only", "--tmpfs", "/tmp:size=64m",
            "--volume", f"{self.root}:{_CONTAINER_WS}:rw",
            "--workdir", _CONTAINER_WS, "--rm",
            img, "sleep", "infinity",
        ], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"docker run failed: {r.stderr}")
        atexit.register(self.stop)
        self.exec(["apt-get", "install", "-y", "-q", "--no-install-recommends",
                   "ripgrep", "tree"], timeout=60, check=False)
        log.info("Sandbox started: %s", self._name)

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", self._name], capture_output=True, timeout=10)

    def safe(self, p: str | Path) -> Path:
        q = Path(str(p))
        if not q.is_absolute(): q = self.root / q
        q = q.resolve()
        if not str(q).startswith(str(self.root)):
            raise PermissionError(f"Path escape blocked: {p!r}")
        return q

    def exec(self, cmd: list[str], cwd: str = ".", timeout: int = 30,
             check: bool = True) -> subprocess.CompletedProcess:
        abs_cwd = self.safe(cwd)
        try:
            rel = abs_cwd.relative_to(self.root)
            ccwd = f"{_CONTAINER_WS}/{rel}" if str(rel) != "." else _CONTAINER_WS
        except ValueError:
            ccwd = _CONTAINER_WS
        return subprocess.run(
            ["docker", "exec", "--workdir", ccwd, self._name] + cmd,
            capture_output=True, text=True, timeout=timeout,
        )

    def cpath(self, p: Path) -> str:
        return f"{_CONTAINER_WS}/{p.relative_to(self.root)}"

    def tree(self, depth: int = 1) -> str:
        lines = [str(self.root)]
        def walk(cur: Path, pre: str, d: int) -> None:
            if d > depth: return
            try: entries = sorted(cur.iterdir(), key=lambda x: (x.is_file(), x.name))
            except: return
            for i, e in enumerate(entries):
                if e.name.startswith(".") or e.name in ("__pycache__", "node_modules", "memory"): continue
                last = i == len(entries) - 1
                lines.append(f"{pre}{'└── ' if last else '├── '}{e.name}")
                if e.is_dir(): walk(e, pre + ("    " if last else "│   "), d + 1)
        walk(self.root, "", 0)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Tools — registered once, shared by all agents
# ══════════════════════════════════════════════════════════════════════════════

TOOLS: dict[str, dict] = {}

def tool(name: str, desc: str, params: dict) -> Callable:
    def dec(fn):
        TOOLS[name] = {"fn": fn, "desc": desc, "params": params}
        return fn
    return dec

def tool_schemas() -> str:
    def schema(n, m):
        req = [k for k,v in m["params"].items() if v.get("required")]
        opt = [k for k,v in m["params"].items() if not v.get("required")]
        args = ", ".join(req) + (f" [{', '.join(opt)}]" if opt else "")
        return f"{n}({args}) — {m['desc']}"
    return "\n".join(schema(n, m) for n, m in TOOLS.items())

def _ok(o: str) -> dict:  return {"ok": True,  "output": o}
def _err(e: str) -> dict: return {"ok": False, "output": f"[ERROR] {e}"}


def register_tools(sb: Sandbox, memory: "Memory", agents: dict[str, "Agent"]) -> None:
    """Register all tools bound to sandbox + memory + agent roster."""

    # ── File I/O (Python-direct on mounted path) ──────────────────────────

    @tool("read_file", "Read a file.",
          {"path": {"required": True}, "start_line": {}, "end_line": {}})
    def read_file(path, start_line=None, end_line=None):
        try: p = sb.safe(path)
        except PermissionError as e: return _err(str(e))
        if not p.is_file(): return _err(f"not a file: {path}")
        lines = p.read_text(errors="replace").splitlines(keepends=True)
        if start_line or end_line:
            lines = lines[int(start_line or 1)-1 : int(end_line) if end_line else None]
        return _ok("".join(lines))

    @tool("write_file", "Write (overwrite) a file.",
          {"path": {"required": True}, "content": {"required": True}})
    def write_file(path, content):
        try: p = sb.safe(path)
        except PermissionError as e: return _err(str(e))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return _ok(f"wrote {path}")

    @tool("patch_file", "Replace first exact match of old_str with new_str.",
          {"path": {"required": True}, "old_str": {"required": True}, "new_str": {"required": True}})
    def patch_file(path, old_str, new_str):
        try: p = sb.safe(path)
        except PermissionError as e: return _err(str(e))
        if not p.is_file(): return _err(f"not found: {path}")
        t = p.read_text()
        n = t.count(old_str)
        if n == 0: return _err("old_str not found")
        if n > 1:  return _err(f"old_str appears {n} times — be more specific")
        p.write_text(t.replace(old_str, new_str, 1))
        return _ok(f"patched {path}")

    @tool("delete_file", "Delete a file.", {"path": {"required": True}})
    def delete_file(path):
        try: p = sb.safe(path)
        except PermissionError as e: return _err(str(e))
        if not p.exists(): return _err(f"not found: {path}")
        p.unlink(); return _ok(f"deleted {path}")

    @tool("list_dir", "Show workspace tree.", {"depth": {}})
    def list_dir(depth=1): return _ok(sb.tree(int(depth)))

    @tool("make_dir", "Create a directory.", {"path": {"required": True}})
    def make_dir(path):
        try: sb.safe(path).mkdir(parents=True, exist_ok=True)
        except PermissionError as e: return _err(str(e))
        return _ok(f"created {path}")

    @tool("move", "Move or rename.", {"src": {"required": True}, "dst": {"required": True}})
    def move(src, dst):
        try: s, d = sb.safe(src), sb.safe(dst)
        except PermissionError as e: return _err(str(e))
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d)); return _ok(f"moved {src} → {dst}")

    # ── Shell + search (docker exec) ──────────────────────────────────────

    BANNED = {"rm", "rmdir", "sudo", "su", "docker", "nsenter"}

    @tool("run_shell", "Run a command in the Docker sandbox. No pipes/redirects.",
          {"command": {"required": True}, "cwd": {}})
    def run_shell(command, cwd="."):
        for ch in "|;&><`$":
            if ch in command: return _err(f"'{ch}' not allowed")
        try: tokens = shlex.split(command)
        except ValueError as e: return _err(str(e))
        if not tokens: return _err("empty command")
        if tokens[0] in BANNED: return _err(f"'{tokens[0]}' not allowed")
        try:
            r = sb.exec(tokens, cwd=cwd, timeout=30)
        except subprocess.TimeoutExpired: return _err("timed out")
        except Exception as e: return _err(str(e))
        out = r.stdout + (f"\n[stderr] {r.stderr.strip()}" if r.returncode != 0 and r.stderr else "")
        return {"ok": r.returncode == 0, "output": out.strip()}

    @tool("search_code", "Search for pattern in files.",
          {"pattern": {"required": True}, "path": {}, "glob": {}})
    def search_code(pattern, path=".", glob=None):
        try: cp = sb.cpath(sb.safe(path))
        except PermissionError as e: return _err(str(e))
        cmd = ["rg", "-n", "--with-filename", "-i", pattern, cp]
        if glob: cmd = cmd[:-2] + ["--glob", glob] + cmd[-2:]
        r = sb.exec(cmd, timeout=15)
        if r.returncode == 127:
            cmd2 = ["grep", "-rni", pattern, cp] + (["--include", glob] if glob else [])
            r = sb.exec(cmd2, timeout=15)
        return _ok(r.stdout.strip() or "no matches")

    # ── Memory tools ──────────────────────────────────────────────────────

    @tool("remember", "Store a fact permanently.", {"fact": {"required": True}})
    def remember(fact): return _ok(memory.store(fact))

    @tool("forget", "Remove facts matching keyword.", {"keyword": {"required": True}})
    def forget(keyword): return _ok(memory.remove(keyword))

    # ── Agent delegation ──────────────────────────────────────────────────

    if agents:
        @tool("ask_agent", "Delegate a task to a specialist agent.",
              {"agent": {"required": True}, "task": {"required": True}})
        def ask_agent(agent, task):
            if agent not in agents:
                return _err(f"unknown agent '{agent}'. Available: {list(agents)}")
            log.info("Delegating to agent '%s': %s", agent, task[:60])

            target = agents[agent]
            answer = target.chat(task)

            # ── Execute COMMAND: lines the sub-agent emitted ───────────────
            # agent.chat() strips everything except the ANSWER: text.
            # Recover the raw LLM output from the last assistant history entry.
            history = getattr(target, "_history", [])
            raw = answer
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    raw = msg.get("content", "")
                    break

            _CMD = re.compile(r"^\s*COMMAND:\s*(.+)$", re.MULTILINE)
            cmds = _CMD.findall(raw)
            cmd_outputs = []

            if cmds:
                log.info("ask_agent: running %d COMMAND(s) emitted by '%s'", len(cmds), agent)
                for cmd in cmds:
                    cmd = cmd.strip()
                    log.info("  exec: %s", cmd)
                    try:
                        result = subprocess.run(
                            cmd, shell=True, capture_output=True,
                            text=True, timeout=30
                        )
                        out = result.stdout.strip()
                        err = result.stderr.strip()
                        rc  = result.returncode
                        log.info("  rc=%d stdout=%r stderr=%r", rc, out[:200], err[:200])
                        summary = f"COMMAND `{cmd}` rc={rc}"
                        if out: summary += f"\nstdout: {out}"
                        if err: summary += f"\nstderr: {err}"
                        cmd_outputs.append(summary)
                    except subprocess.TimeoutExpired:
                        cmd_outputs.append(f"COMMAND `{cmd}` timed out")
                    except Exception as exc:
                        cmd_outputs.append(f"COMMAND `{cmd}` error: {exc}")

                # Attach results to the sub-agent's last assistant message so
                # it can reason about them if called again this session.
                for i in range(len(history) - 1, -1, -1):
                    if history[i].get("role") == "assistant":
                        history[i] = dict(history[i])
                        history[i]["content"] += (
                            "\n\nCOMMAND_QUEUE_RESULTS:\n" + "\n\n".join(cmd_outputs)
                        )
                        break

            combined = answer
            if cmd_outputs:
                combined += "\n\nExecution results:\n" + "\n\n".join(cmd_outputs)
            return _ok(combined)


# ══════════════════════════════════════════════════════════════════════════════
#  Memory — git-backed, progressive disclosure
# ══════════════════════════════════════════════════════════════════════════════

class Memory:
    SUMMARY_LINES = 6

    def __init__(self, root: Path) -> None:
        self.root  = root
        self.facts = root / "facts.md"
        self._sessions = root / "sessions"

    def init(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._sessions.mkdir(exist_ok=True)
        if not (self.root / ".git").exists():
            self._git("init", "-q")
            self._git("commit", "--allow-empty", "-m", "init memory repo")
        if not self.facts.exists():
            self.facts.write_text("# Memory\n\n")
            self._commit("init: facts.md")

    def summary(self) -> str:
        bullets = self._bullets()
        if not bullets: return ""
        lines = bullets[:self.SUMMARY_LINES]
        s = "MEMORY:\n" + "\n".join(f"• {l}" for l in lines)
        if len(bullets) > self.SUMMARY_LINES:
            s += f"\n• … ({len(bullets)-self.SUMMARY_LINES} more in memory/facts.md)"
        return s + "\n"

    def reflect(self, user: str, assistant: str) -> None:
        """Extract facts heuristically from one turn — no LLM call."""
        new = []
        text = user + " " + assistant
        for pat, tmpl in [
            (r"\b(arch linux|ubuntu|debian|fedora|nixos)\b", "OS: {}"),
            (r"\b(i3wm|sway|hyprland|dwm)\b",               "WM: {}"),
            (r"\bpython\s*(\d+\.\d+)\b",                    "Python: {}"),
            (r"\bcuda\s*(\d+\.\d+)\b",                      "CUDA: {}"),
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m: new.append(tmpl.format(m.group(1).lower()))

        for p in re.findall(r"I (?:prefer|always use|use) ([^.,;!?]{3,40})", text, re.IGNORECASE)[:2]:
            new.append(f"prefers: {p.strip().lower()}")
        for p in re.findall(r"~/[\w/.\-]+|/home/\w+/[\w/.\-]+", text)[:2]:
            new.append(f"path: {p}")

        existing = set(self._bullets())
        truly_new = [f for f in new if f not in existing]
        if truly_new:
            self.facts.write_text(
                self.facts.read_text().rstrip("\n") +
                "\n" + "\n".join(f"• {f}" for f in truly_new) + "\n"
            )
            self._commit(f"reflect: {truly_new[0][:40]}")

    def store(self, fact: str) -> str:
        self.facts.write_text(self.facts.read_text().rstrip("\n") + f"\n• {fact}\n")
        self._commit(f"remember: {fact[:50]}")
        return f"Remembered: {fact}"

    def remove(self, keyword: str) -> str:
        lines = self.facts.read_text().splitlines(keepends=True)
        kept = [l for l in lines if keyword.lower() not in l.lower()]
        removed = len(lines) - len(kept)
        self.facts.write_text("".join(kept))
        if removed: self._commit(f"forget: '{keyword}'")
        return f"Removed {removed} line(s)."

    def save_session(self, history: list[dict]) -> None:
        today = date.today().isoformat()
        path  = self._sessions / f"{today}.md"
        turns = [f"**{m['role']}**: {m['content'][:200]}"
                 for m in history if m.get("role") in ("user","assistant")
                 and not m.get("content","").startswith("TOOL_RESULT")]
        if not turns: return
        existing = path.read_text() if path.exists() else f"# {today}\n\n"
        path.write_text(existing + "\n---\n" + "\n".join(turns[:20]) + "\n")
        self._commit(f"session: {today}")

    def _bullets(self) -> list[str]:
        if not self.facts.exists(): return []
        return [l.lstrip("•- ").strip() for l in self.facts.read_text().splitlines()
                if l.strip() and not l.startswith("#")]

    def _git(self, *args: str) -> None:
        subprocess.run(["git", "-C", str(self.root)] + list(args),
                       capture_output=True, text=True)

    def _commit(self, msg: str) -> None:
        self._git("add", "-A")
        self._git("commit", "-m", msg)


# ══════════════════════════════════════════════════════════════════════════════
#  Agent — one LLM server + system prompt + shared tools + own history
# ══════════════════════════════════════════════════════════════════════════════

_TOOL_RE   = re.compile(r"TOOL:\s*(\w+)\s*\nARGS:\s*(\{.*?\})", re.DOTALL)
_ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.DOTALL)

_FEW_SHOT = """
EXAMPLES:
User: list files
TOOL: run_shell
ARGS: {"command": "ls -la"}
TOOL_RESULT (run_shell): main.py  config.yaml
ANSWER: The workspace has main.py and config.yaml.

User: read main.py
TOOL: read_file
ARGS: {"path": "main.py"}
TOOL_RESULT (read_file): print("hello")
ANSWER: main.py has one line: print hello.

User: hello
ANSWER: Hello! What can I do for you?

User: write a python file and run it
TOOL: ask_agent
ARGS: {"agent": "coder", "task": "write /workspace/hello.py containing: print('hello')"}
TOOL_RESULT (ask_agent): wrote /workspace/hello.py. Execution results: COMMAND rc=0 stdout: hello
TOOL: ask_agent
ARGS: {"agent": "runner", "task": "run /workspace/hello.py"}
TOOL_RESULT (ask_agent): stdout: hello / stderr: none
ANSWER: Created hello.py and ran it. Output: hello
"""

class Agent:
    def __init__(self, llm, system: str, memory: Memory,
                 max_iter: int = 8, name: str = "agent") -> None:
        self._llm     = llm
        self._system  = system
        self._memory  = memory
        self._max     = max_iter
        self._history: list[dict] = []
        self.name     = name

    def _build_system(self) -> str:
        return (
            self._system + "\n\n"
            + self._memory.summary()
            + "\nTOOLS (required, [optional]):\n"
            + tool_schemas() + "\n"
            + "RULES: ONLY output TOOL:/ARGS: or ANSWER:. Never explain. One tool at a time.\n"
            + _FEW_SHOT
        )

    def chat(self, user_text: str) -> str:
        log.info("[%s] User: %s", self.name, user_text)
        self._history.append({"role": "user", "content": user_text})

        for i in range(self._max):
            messages = [{"role": "system", "content": self._build_system()}] + self._history
            raw = self._llm.generate(messages)
            log.debug("[%s] raw[%d]: %s", self.name, i, raw[:150])

            m = _ANSWER_RE.search(raw)
            if m:
                answer = m.group(1).strip()
                self._history.append({"role": "assistant", "content": raw})
                self._memory.reflect(user_text, answer)
                self._trim()
                return answer

            m = _TOOL_RE.search(raw)
            if m:
                tname = m.group(1).strip()
                try:
                    args = json.loads(m.group(2))
                    result = self._run_tool(tname, args)
                except Exception as e:
                    result = f"[ERROR] {e}"
                self._history.append({"role": "assistant", "content": raw.strip()})
                self._history.append({"role": "user", "content": f"TOOL_RESULT ({tname}): {result}"})
                continue

            # No match — treat as answer
            self._history.append({"role": "assistant", "content": raw})
            self._memory.reflect(user_text, raw)
            self._trim()
            return raw.strip()

        return "Hit step limit — try /reset or a simpler request."

    def reset(self) -> None:
        self._memory.save_session(self._history)
        self._history.clear()

    def _run_tool(self, name: str, args: dict) -> str:
        if name not in TOOLS:
            return f"[ERROR] unknown tool '{name}'. Available: {list(TOOLS)}"
        try:
            return TOOLS[name]["fn"](**args)["output"]
        except TypeError as e:
            return f"[ERROR] wrong args: {e}"
        except Exception as e:
            log.exception("Tool %s crashed", name)
            return f"[ERROR] {e}"

    def _trim(self) -> None:
        if len(self._history) > 20:
            self._history = self._history[-20:]


# ══════════════════════════════════════════════════════════════════════════════
#  Router — picks which agent handles a request
# ══════════════════════════════════════════════════════════════════════════════

class Router:
    """
    Classifies each request and delegates to the appropriate specialist Agent.
    Uses few-shot prompting + fuzzy matching to handle LLM chattiness.
    """

    def __init__(self, router_llm, agents: dict[str, Agent]) -> None:
        self._llm    = router_llm
        self._agents = agents

    def chat(self, user_text: str) -> str:
        agent_name = self._route(user_text)
        log.info("Router → %s", agent_name)
        agent = self._agents.get(agent_name) or next(iter(self._agents.values()))
        self._last_agent = agent          # expose for cli.py history recovery
        return agent.chat(user_text)

    def reset(self) -> None:
        for a in self._agents.values():
            a.reset()

    def _route(self, text: str) -> str:
        names = list(self._agents.keys())

        # Build a few-shot prompt using the actual agent names so the LLM
        # has concrete examples of the exact output format expected.
        examples = "\n".join(
            f"Request: do something with {n} → {n}" for n in names
        )

        msgs = [
            {
                "role": "user",
                "content": (
                    f"You are a router. Output ONLY one agent name from this list: "
                    f"{', '.join(names)}\n"
                    f"No explanation. No punctuation. One word.\n\n"
                    f"Examples:\n{examples}\n\n"
                    f"Request: {text}\n"
                    f"Agent:"
                ),
            },
            # Prime the assistant turn with an empty string so the model
            # completes a name rather than starting a new sentence.
            {
                "role": "assistant",
                "content": "",
            },
        ]

        raw = self._llm.generate(msgs, stop=["\n", " ", ".", ","]).strip().lower()
        raw = re.sub(r"[^a-z_]", "", raw)
        log.info("Router raw LLM output: %r", raw)

        # Exact match
        if raw in self._agents:
            log.info("Router exact match → '%s'", raw)
            return raw

        # Fuzzy match — handles "plan" → "planner", "cod" → "coder" etc.
        for name in names:
            if raw in name or name in raw:
                log.info("Router fuzzy match %r → '%s'", raw, name)
                return name

        # Nothing matched — fall back to first agent
        fallback = next(iter(self._agents))
        log.warning("Router unrecognised %r — falling back to '%s'", raw, fallback)
        return fallback


# ══════════════════════════════════════════════════════════════════════════════
#  Factory — builds everything from config dict
# ══════════════════════════════════════════════════════════════════════════════

def build(cfg: dict) -> tuple[dict[str, Agent], Router | None]:
    """
    Build all agents from config. Returns (agents_dict, router_or_None).

    agents_dict keys are agent names from config.yaml.
    router is None if no 'router' agent is defined in config.
    """
    from inference import LLM

    workspace = Path(cfg["workspace"]).expanduser().resolve()
    servers_cfg = cfg.get("servers", {"default": {"url": "http://localhost:8080"}})
    agents_cfg  = cfg.get("agents",  {"default": {"server": "default", "system": "You are a helpful assistant."}})

    # Build LLM clients (one per unique server URL)
    llms: dict[str, LLM] = {}
    for sname, scfg in servers_cfg.items():
        llms[sname] = LLM(scfg)

    # Shared memory + sandbox
    memory = Memory(workspace / "memory")
    memory.init()

    sandbox = Sandbox(workspace, cfg.get("docker", {}))
    sandbox.start()

    # Build all agents first (without tools — tools need agent refs for ask_agent)
    agents: dict[str, Agent] = {}
    for aname, acfg in agents_cfg.items():
        if aname == "router":
            continue  # router is built separately
        server_name = acfg.get("server", "default")
        llm = llms.get(server_name) or list(llms.values())[0]
        system = acfg["system"].format(
            workspace=workspace,
            agent_names=", ".join(k for k in agents_cfg if k != "router"),
        )
        agents[aname] = Agent(llm=llm, system=system, memory=memory, name=aname)

    # Register tools now that we have the agent roster
    register_tools(sandbox, memory, agents)

    # Build router if defined
    router = None
    if "router" in agents_cfg:
        rcfg   = agents_cfg["router"]
        rllm   = llms.get(rcfg.get("server", "default")) or list(llms.values())[0]
        rsys   = rcfg["system"].format(
            workspace=workspace,
            agent_names=", ".join(agents.keys()),
        )
        router_agent = Agent(llm=rllm, system=rsys, memory=memory, name="router")
        router = Router(router_llm=rllm, agents=agents)

    return agents, router
