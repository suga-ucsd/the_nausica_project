"""
memory.py — git-backed context repository for Nausicaa.

Inspired by Letta's Context Repositories blog post (Feb 2026).

Design for TinyLlama's 2048-token constraint
─────────────────────────────────────────────
Token budget:
  system prompt + tools + examples  ≈  620 tokens
  memory summary (injected below)   ≈  60–80 tokens   ← this file controls this
  conversation history              ≈  1300 tokens remaining

Progressive disclosure (the key idea from the blog):
  • SHORT bullet-point summary always in system prompt  (~60 tokens)
  • Full memory files sit in memory/ on disk
  • Agent can read_file memory/facts.md for full details on demand
  • File tree in system prompt acts as navigation signal

Memory files (all plain markdown, git-tracked):
  memory/
    facts.md          ← key facts about user/project, concise
    sessions/
      YYYY-MM-DD.md   ← per-session summaries

Reflection (no extra LLM call — heuristics only):
  After each turn, scan the conversation for:
    • Names, usernames, system info (arch, distro, GPU)
    • File paths and project names mentioned
    • Errors and how they were fixed
    • Explicit preferences ("I prefer", "I always", "I use")
  Append new facts to memory/facts.md.
  Git-commit with an informative message.

Git-backing:
  Every write is a git commit. This means:
    • Full history of what the agent has learned
    • Easy rollback if memory gets corrupted
    • Ready for future multi-agent/subagent merges (as blog describes)
"""
from __future__ import annotations

import logging
import re
import subprocess
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

# How many lines from facts.md to inject into system prompt.
# Each line is ~10-15 tokens. 6 lines ≈ 60-90 tokens.
SUMMARY_LINES = 6


class MemoryStore:
    """
    Git-backed flat-file memory for a Nausicaa agent session.

    Usage:
        mem = MemoryStore(workspace / "memory")
        mem.init()

        # In Agent.__init__:
        system_prompt += mem.summary_block()

        # After each Agent.chat() turn:
        mem.reflect(user_text, assistant_answer)

        # At session end:
        mem.save_session(history)
    """

    def __init__(self, memory_dir: Path) -> None:
        self.root     = memory_dir
        self.facts    = memory_dir / "facts.md"
        self.sessions = memory_dir / "sessions"

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def init(self) -> None:
        """Create directory structure and git repo if not already present."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions.mkdir(exist_ok=True)

        if not (self.root / ".git").exists():
            self._git("init", "-q")
            self._git("commit", "--allow-empty", "-m", "init: create memory repository")
            log.info("Memory repository initialised at %s", self.root)

        if not self.facts.exists():
            self.facts.write_text(
                "# Nausicaa Memory\n"
                "<!-- Agent-maintained facts. Concise bullets only. -->\n\n"
            )
            self._commit_file(self.facts, "init: create facts.md")

    # ── System prompt injection ───────────────────────────────────────────

    def summary_block(self) -> str:
        """
        Returns a short block to append to the system prompt.
        Kept deliberately small — SUMMARY_LINES bullets ≈ 60-90 tokens.

        Full memory is always available via:  read_file memory/facts.md
        """
        bullets = self._load_bullets()
        if not bullets:
            return ""

        lines = bullets[:SUMMARY_LINES]
        block = "MEMORY (what I know about you and this project):\n"
        block += "\n".join(f"• {l}" for l in lines)
        if len(bullets) > SUMMARY_LINES:
            block += f"\n• … ({len(bullets) - SUMMARY_LINES} more in memory/facts.md)"
        return "\n" + block + "\n"

    # ── Reflection — runs after each turn ─────────────────────────────────

    def reflect(self, user: str, assistant: str) -> None:
        """
        Extract facts from one conversation turn and persist them.
        Heuristic-based — no extra LLM call needed.
        """
        new_facts = []

        combined = user + " " + assistant

        # ── System / environment info ─────────────────────────────────
        _env_patterns = [
            (r"\b(arch linux|ubuntu|debian|fedora|nixos|manjaro)\b", "OS: {}"),
            (r"\b(i3wm|i3 window manager|sway|hyprland|dwm)\b",    "WM: {}"),
            (r"\b(\d+(?:\.\d+)?\s*gb\s*(?:vram|gpu))\b",           "GPU VRAM: {}"),
            (r"\bpython\s*(\d+\.\d+)\b",                            "Python: {}"),
            (r"\bcuda\s*(\d+\.\d+)\b",                              "CUDA: {}"),
        ]
        for pattern, template in _env_patterns:
            m = re.search(pattern, combined, re.IGNORECASE)
            if m:
                fact = template.format(m.group(1).lower())
                new_facts.append(fact)

        # ── Usernames / names ─────────────────────────────────────────
        m = re.search(r"\bmy name is (\w+)\b|\bI(?:'m| am) (\w+)\b", combined, re.IGNORECASE)
        if m:
            name = m.group(1) or m.group(2)
            new_facts.append(f"user name: {name}")

        # ── Explicit preferences ──────────────────────────────────────
        pref_m = re.findall(
            r"I (?:prefer|always use|like to use|want to use|use) ([^.,;!?]{3,40})",
            combined, re.IGNORECASE
        )
        for p in pref_m[:2]:
            new_facts.append(f"prefers: {p.strip().lower()}")

        # ── Project / file paths mentioned ────────────────────────────
        path_m = re.findall(r"~/[\w/.\-]+|/home/\w+/[\w/.\-]+", combined)
        for p in path_m[:2]:
            new_facts.append(f"project path: {p}")

        # ── Errors and fixes ──────────────────────────────────────────
        if re.search(r"\berror\b|\bfixed\b|\bsolved\b|\bworking now\b", combined, re.IGNORECASE):
            # Capture a short snippet around the error for context
            m = re.search(r"(?:error|fixed|solved)[:\s]+([^.!?\n]{10,80})", combined, re.IGNORECASE)
            if m:
                new_facts.append(f"note: {m.group(1).strip().lower()[:80]}")

        # ── Deduplicate and append only genuinely new facts ───────────
        existing = self._load_bullets()
        existing_lower = {e.lower() for e in existing}

        truly_new = [
            f for f in new_facts
            if f.lower() not in existing_lower
            and not any(f.lower() in e.lower() or e.lower() in f.lower()
                        for e in existing_lower)
        ]

        if truly_new:
            self._append_facts(truly_new)
            self._commit_file(
                self.facts,
                f"reflect: learned {len(truly_new)} fact(s) — {truly_new[0][:40]}"
            )
            log.debug("Memory: +%d facts: %s", len(truly_new), truly_new)

    # ── Session summary ───────────────────────────────────────────────────

    def save_session(self, history: list[dict]) -> None:
        """
        Write a brief session log to memory/sessions/YYYY-MM-DD.md.
        Captures the conversation turns in compressed form.
        """
        today = date.today().isoformat()
        path  = self.sessions / f"{today}.md"

        turns = []
        for msg in history:
            role    = msg.get("role", "")
            content = msg.get("content", "")[:200]   # truncate long tool results
            if role in ("user", "assistant") and not content.startswith("TOOL_RESULT"):
                turns.append(f"**{role}**: {content}")

        if not turns:
            return

        existing = path.read_text() if path.exists() else f"# Session {today}\n\n"
        separator = "\n---\n"
        new_block = "\n".join(turns[:20])   # cap at 20 turns
        path.write_text(existing + separator + new_block + "\n")
        self._commit_file(path, f"session: {today} — {len(turns)} turns")

    # ── Manual memory tool (exposed to agent via tools.py) ────────────────

    def remember(self, fact: str) -> str:
        """
        Explicitly store a fact. Called by the remember() tool.
        Returns confirmation string.
        """
        self._append_facts([fact.strip()])
        self._commit_file(self.facts, f"remember: {fact[:50]}")
        return f"Remembered: {fact}"

    def forget(self, keyword: str) -> str:
        """
        Remove lines from facts.md that contain the keyword.
        """
        lines     = self.facts.read_text().splitlines(keepends=True)
        kept      = [l for l in lines if keyword.lower() not in l.lower()]
        removed   = len(lines) - len(kept)
        if removed:
            self.facts.write_text("".join(kept))
            self._commit_file(self.facts, f"forget: removed {removed} line(s) matching '{keyword}'")
        return f"Removed {removed} line(s) matching '{keyword}'."

    # ── Git helpers ───────────────────────────────────────────────────────

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.root)] + list(args),
            capture_output=True, text=True,
        )

    def _commit_file(self, path: Path, message: str) -> None:
        self._git("add", str(path.relative_to(self.root)))
        r = self._git("commit", "-m", message)
        if r.returncode != 0 and "nothing to commit" not in r.stdout:
            log.debug("git commit: %s", r.stderr.strip()[:100])

    # ── Internal helpers ──────────────────────────────────────────────────

    def _load_bullets(self) -> list[str]:
        """Load non-empty, non-header lines from facts.md."""
        if not self.facts.exists():
            return []
        lines = self.facts.read_text().splitlines()
        return [
            l.lstrip("•- ").strip()
            for l in lines
            if l.strip() and not l.startswith("#") and not l.startswith("<!--")
        ]

    def _append_facts(self, facts: list[str]) -> None:
        current = self.facts.read_text() if self.facts.exists() else ""
        addition = "\n".join(f"• {f}" for f in facts) + "\n"
        self.facts.write_text(current.rstrip("\n") + "\n" + addition)
