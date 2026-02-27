"""
tools.py — sandbox, all tools, and the ReAct agent loop.

Docker IS the sandbox. Shell commands run via `docker exec`.
File I/O is Python-direct on the host-side mount (faster, equally safe).
Path validation happens on the host before any disk operation.
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
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

WORKSPACE_IN_CONTAINER = "/workspace"

# ══════════════════════════════════════════════════════════════════════════════
#  Docker sandbox
# ══════════════════════════════════════════════════════════════════════════════

class Sandbox:
    """
    Starts a Docker container with the workspace mounted at /workspace.
    Provides:
      safe(path)  → validates host path stays inside workspace root
      exec(cmd)   → runs command inside the container via docker exec
      All file ops use Python-direct I/O on the mounted path.
    """

    def __init__(self, workspace: Path, docker_cfg: dict) -> None:
        self.root = workspace.expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"Workspace not found: {self.root}")
        if not shutil.which("docker"):
            raise RuntimeError("docker not found. Install docker and add yourself to the docker group.")

        self._cfg  = docker_cfg
        self._name = f"nausicaa_{uuid.uuid4().hex[:8]}"
        self._id:  str | None = None

    def start(self) -> None:
        img = self._cfg.get("image", "debian:bookworm-slim")
        log.info("Starting sandbox container %s (image=%s)", self._name, img)
        subprocess.run(["docker", "pull", "-q", img], timeout=120, check=False)
        r = subprocess.run([
            "docker", "run", "--detach",
            "--name",         self._name,
            "--network",      self._cfg.get("network",  "none"),
            "--memory",       self._cfg.get("memory",   "512m"),
            "--cpus",         str(self._cfg.get("cpus", "1.0")),
            "--pids-limit",   str(self._cfg.get("pids",  128)),
            "--security-opt", "no-new-privileges:true",
            "--read-only",
            "--tmpfs",        "/tmp:size=64m",
            "--volume",       f"{self.root}:{WORKSPACE_IN_CONTAINER}:rw",
            "--workdir",      WORKSPACE_IN_CONTAINER,
            img, "sleep", "infinity",
        ], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"docker run failed:\n{r.stderr}")
        self._id = r.stdout.strip()[:12]
        atexit.register(self.stop)
        # best-effort tool install (no-op if network=none)
        self.exec(["apt-get", "install", "-y", "-q", "--no-install-recommends",
                   "ripgrep", "tree", "diffutils"], timeout=60, check=False)
        log.info("Sandbox ready (id=%s)", self._id)

    def stop(self) -> None:
        if not self._id: return
        subprocess.run(["docker", "rm", "-f", self._name], capture_output=True, timeout=10)
        self._id = None

    # ── path guard ────────────────────────────────────────────────────────

    def safe(self, p: str | Path) -> Path:
        """Resolve and verify path stays inside workspace root. Raises on escape."""
        q = Path(str(p))
        if not q.is_absolute():
            q = self.root / q
        q = q.resolve()
        if not str(q).startswith(str(self.root)):
            raise PermissionError(f"Path escape blocked: {p!r} → {q}")
        return q

    # ── docker exec ───────────────────────────────────────────────────────

    def exec(self, cmd: list[str], cwd: str = ".", timeout: int = 30,
             check: bool = True) -> subprocess.CompletedProcess:
        abs_cwd = self.safe(cwd)
        try:
            rel = abs_cwd.relative_to(self.root)
            container_cwd = f"{WORKSPACE_IN_CONTAINER}/{rel}" if str(rel) != "." else WORKSPACE_IN_CONTAINER
        except ValueError:
            container_cwd = WORKSPACE_IN_CONTAINER

        return subprocess.run(
            ["docker", "exec", "--workdir", container_cwd, self._name] + cmd,
            capture_output=True, text=True, timeout=timeout,
        )

    def c_path(self, host_path: Path) -> str:
        """Translate host absolute path → container path string."""
        try:
            rel = host_path.relative_to(self.root)
            return f"{WORKSPACE_IN_CONTAINER}/{rel}"
        except ValueError:
            raise PermissionError(f"{host_path} is outside workspace")

    def tree(self, depth: int = 1) -> str:
        lines = [str(self.root)]
        def walk(cur: Path, prefix: str, d: int) -> None:
            if d > depth: return
            try:
                entries = sorted(cur.iterdir(), key=lambda p: (p.is_file(), p.name))
            except PermissionError: return
            for i, e in enumerate(entries):
                if e.name.startswith(".") or e.name in ("__pycache__", "node_modules"): continue
                last = i == len(entries) - 1
                lines.append(f"{prefix}{'└── ' if last else '├── '}{e.name}")
                if e.is_dir(): walk(e, prefix + ("    " if last else "│   "), d+1)
        walk(self.root, "", 0)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Tools  (each returns {"ok": bool, "output": str})
# ══════════════════════════════════════════════════════════════════════════════

def _ok(output: str) -> dict:  return {"ok": True,  "output": output}
def _err(msg: str)   -> dict:  return {"ok": False, "output": f"[ERROR] {msg}"}

TOOLS: dict[str, dict] = {}  # filled by @tool decorator

def tool(name: str, desc: str, params: dict):
    """Decorator that registers a function as a tool."""
    def decorator(fn):
        TOOLS[name] = {"fn": fn, "desc": desc, "params": params}
        return fn
    return decorator

def schema_line(name: str, meta: dict) -> str:
    req = [k for k,v in meta["params"].items() if v.get("required")]
    opt = [k for k,v in meta["params"].items() if not v.get("required")]
    args = ", ".join(req) + (f" [{', '.join(opt)}]" if opt else "")
    return f"{name}({args}) — {meta['desc']}"


# ── File ops ──────────────────────────────────────────────────────────────────

def _make_file_tools(sb: Sandbox):

    @tool("read_file", "Read a file. Optional line range.",
          {"path": {"required": True}, "start_line": {}, "end_line": {}})
    def read_file(path, start_line=None, end_line=None):
        try:
            p = sb.safe(path)
        except PermissionError as e: return _err(str(e))
        if not p.is_file(): return _err(f"Not a file: {path}")
        lines = p.read_text(errors="replace").splitlines(keepends=True)
        if start_line or end_line:
            lines = lines[(int(start_line or 1)-1):(int(end_line) if end_line else None)]
        return _ok("".join(lines))

    @tool("write_file", "Write (overwrite) a file with new content.",
          {"path": {"required": True}, "content": {"required": True}})
    def write_file(path, content):
        try:
            p = sb.safe(path)
        except PermissionError as e: return _err(str(e))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return _ok(f"Wrote {path}")

    @tool("patch_file", "Replace first exact occurrence of old_str with new_str.",
          {"path": {"required": True}, "old_str": {"required": True}, "new_str": {"required": True}})
    def patch_file(path, old_str, new_str):
        try:
            p = sb.safe(path)
        except PermissionError as e: return _err(str(e))
        if not p.is_file(): return _err(f"Not found: {path}")
        txt = p.read_text()
        n = txt.count(old_str)
        if n == 0: return _err("old_str not found")
        if n > 1:  return _err(f"old_str appears {n} times — be more specific")
        p.write_text(txt.replace(old_str, new_str, 1))
        return _ok(f"Patched {path}")

    @tool("delete_file", "Permanently delete a file.",
          {"path": {"required": True}})
    def delete_file(path):
        try:
            p = sb.safe(path)
        except PermissionError as e: return _err(str(e))
        if not p.exists(): return _err(f"Not found: {path}")
        p.unlink()
        return _ok(f"Deleted {path}")

    @tool("list_dir", "Show workspace directory tree.",
          {"depth": {}})
    def list_dir(depth=1):
        return _ok(sb.tree(depth=int(depth)))

    @tool("make_dir", "Create a directory.",
          {"path": {"required": True}})
    def make_dir(path):
        try:
            p = sb.safe(path)
        except PermissionError as e: return _err(str(e))
        p.mkdir(parents=True, exist_ok=True)
        return _ok(f"Created {path}")

    @tool("move", "Move or rename a file or directory.",
          {"src": {"required": True}, "dst": {"required": True}})
    def move(src, dst):
        try:
            s, d = sb.safe(src), sb.safe(dst)
        except PermissionError as e: return _err(str(e))
        if not s.exists(): return _err(f"Source not found: {src}")
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        return _ok(f"Moved {src} → {dst}")


# ── Shell — runs inside Docker container ──────────────────────────────────────

def _make_shell_tool(sb: Sandbox, whitelist: list[str]):
    BANNED = {"rm", "rmdir", "sudo", "su", "docker", "nsenter"}
    SHELL_CHARS = set("|;&><`$")
    allowed = set(whitelist) if whitelist else None

    @tool("run_shell", "Run a command inside the Docker sandbox. No pipes/redirects.",
          {"command": {"required": True}, "cwd": {}})
    def run_shell(command, cwd="."):
        for ch in SHELL_CHARS:
            if ch in command:
                return _err(f"Shell character '{ch}' not allowed")
        try:
            tokens = shlex.split(command)
        except ValueError as e:
            return _err(f"Parse error: {e}")
        if not tokens: return _err("Empty command")
        if tokens[0] in BANNED:
            return _err(f"'{tokens[0]}' is not allowed")
        if allowed and tokens[0] not in allowed:
            return _err(f"'{tokens[0]}' not in whitelist: {sorted(allowed)}")
        try:
            r = sb.exec(tokens, cwd=cwd, timeout=30)
        except subprocess.TimeoutExpired:
            return _err("Timed out after 30s")
        except Exception as e:
            return _err(f"exec error: {e}")
        out = r.stdout + (("\n[stderr] " + r.stderr.strip()) if r.returncode != 0 and r.stderr else "")
        return {"ok": r.returncode == 0, "output": out.strip()}


# ── Search — runs inside Docker container ─────────────────────────────────────

def _make_search_tools(sb: Sandbox):

    @tool("search_code", "Search for pattern in source files.",
          {"pattern": {"required": True}, "path": {}, "glob": {}})
    def search_code(pattern, path=".", glob=None):
        try:
            cp = sb.c_path(sb.safe(path))
        except PermissionError as e: return _err(str(e))
        cmd = ["rg", "--line-number", "--with-filename", "-i", pattern, cp]
        if glob: cmd = cmd[:-2] + ["--glob", glob] + cmd[-2:]
        r = sb.exec(cmd, timeout=15)
        if r.returncode == 127:  # rg not found, try grep
            cmd = ["grep", "-rni", pattern, cp]
            if glob: cmd += ["--include", glob]
            r = sb.exec(cmd, timeout=15)
        return _ok(r.stdout.strip() or "No matches") if r.returncode in (0, 1) else _err(r.stderr)

    @tool("find_file", "Find files by name pattern.",
          {"name": {"required": True}, "path": {}})
    def find_file(name, path="."):
        try:
            cp = sb.c_path(sb.safe(path))
        except PermissionError as e: return _err(str(e))
        r = sb.exec(["find", cp, "-name", name, "-type", "f"], timeout=10)
        out = r.stdout.replace(WORKSPACE_IN_CONTAINER + "/", "").replace(WORKSPACE_IN_CONTAINER, ".").strip()
        return _ok(out or "No files found")


def build_tools(sb: Sandbox, whitelist: list) -> None:
    """Register all tools bound to this sandbox instance."""
    _make_file_tools(sb)
    _make_shell_tool(sb, whitelist)
    _make_search_tools(sb)


# ══════════════════════════════════════════════════════════════════════════════
#  ReAct Agent loop
# ══════════════════════════════════════════════════════════════════════════════

_TOOL_RE   = re.compile(r"TOOL:\s*(\w+)\s*\nARGS:\s*(\{.*?\})", re.DOTALL)
_ANSWER_RE = re.compile(r"ANSWER:\s*(.+)",                        re.DOTALL)

SYSTEM_PROMPT = """\
You are Nausicaa, a voice coding assistant. Help the user work with code and files.
WORKSPACE: {workspace}
Answers are spoken aloud — be brief and direct.

TOOLS (required, [optional]):
{schemas}

RULES — follow these exactly:
1. NEVER write code or scripts as an answer. USE a tool instead.
2. ONLY output TOOL:/ARGS: or ANSWER:. Nothing else.
3. One tool call at a time.

EXAMPLES:
User: list files
TOOL: run_shell
ARGS: {{"command": "ls -la"}}
TOOL_RESULT (run_shell): main.py  config.yaml
ANSWER: The workspace has main.py and config.yaml.

User: read main.py
TOOL: read_file
ARGS: {{"path": "main.py"}}
TOOL_RESULT (read_file): print("hello")
ANSWER: main.py has one line: print hello.

User: search for TODO
TOOL: run_shell
ARGS: {{"command": "grep -rn TODO ."}}
TOOL_RESULT (run_shell): utils.py:12:# TODO fix this
ANSWER: Found one TODO in utils.py at line 12.

User: hello
ANSWER: Hello! What would you like me to do?

WORKSPACE FILES:
{tree}
"""


class Agent:
    def __init__(self, llm, workspace: Path, docker_cfg: dict,
                 whitelist: list, max_iter: int = 8) -> None:
        self._llm = llm
        self._max = max_iter
        self._history: list[dict] = []  # running message list

        # Start sandbox + register tools
        self._sb = Sandbox(workspace, docker_cfg)
        self._sb.start()
        build_tools(self._sb, whitelist)

        schemas = "\n".join(schema_line(n, m) for n, m in TOOLS.items())
        self._system = SYSTEM_PROMPT.format(
            workspace=workspace,
            schemas=schemas,
            tree=self._sb.tree(depth=1),
        )

    def chat(self, user_text: str) -> str:
        log.info("User: %s", user_text)
        self._history.append({"role": "user", "content": user_text})

        for i in range(self._max):
            messages = [{"role": "system", "content": self._system}] + self._history
            raw = self._llm.generate(messages)
            log.debug("LLM [%d]: %s", i, raw[:200])

            # Final answer?
            m = _ANSWER_RE.search(raw)
            if m:
                answer = m.group(1).strip()
                self._history.append({"role": "assistant", "content": raw})
                self._trim_history()
                log.info("Answer: %s", answer)
                return answer

            # Tool call?
            m = _TOOL_RE.search(raw)
            if m:
                tool_name = m.group(1).strip()
                try:
                    args = json.loads(m.group(2))
                except json.JSONDecodeError as e:
                    result = f"[ERROR] Bad JSON: {e}"
                else:
                    result = self._run_tool(tool_name, args)

                # Inject result into history as assistant + tool_result pair
                self._history.append({"role": "assistant", "content": raw.strip()})
                self._history.append({
                    "role": "user",
                    "content": f"TOOL_RESULT ({tool_name}): {result}"
                })
                continue

            # Neither — treat raw output as answer
            log.warning("No TOOL or ANSWER found — treating as answer")
            self._history.append({"role": "assistant", "content": raw})
            self._trim_history()
            return raw.strip()

        return "I hit the step limit. Please try a simpler request or type /reset."

    def reset(self) -> None:
        self._history.clear()

    def _run_tool(self, name: str, args: dict) -> str:
        if name not in TOOLS:
            return f"[ERROR] Unknown tool '{name}'. Available: {list(TOOLS)}"
        try:
            result = TOOLS[name]["fn"](**args)
            return result["output"]
        except TypeError as e:
            return f"[ERROR] Wrong arguments: {e}"
        except Exception as e:
            log.exception("Tool %s crashed", name)
            return f"[ERROR] {e}"

    def _trim_history(self) -> None:
        # Keep last 10 turns to stay within context window
        if len(self._history) > 20:
            self._history = self._history[-20:]
