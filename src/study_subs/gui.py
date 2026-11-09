"""영어 듣기 학습 GUI.

왼쪽 문장 목록에서 항목을 고르고 Space로 해당 구간을 재생하며,
Enter로 오른쪽 화면의 한국어 번역을 보이거나 숨깁니다.
"""

from __future__ import annotations

import argparse
import io
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox
from typing import Callable

import customtkinter as ctk

from .audio import find_ffmpeg
from .project import (
    Project,
    create_project,
    default_project_root,
    discover_projects,
    load_project,
    save_project_cues,
)
from .subtitles import format_mmss
from .tail_fix import CueSpan, TailFix, analyze_tail_clipping
from .ui_state import load_last_index, save_last_index


LIGHT_THEME: dict[str, str] = {
    "window": "#f1f5f9",
    "header_bg": "#0f172a",
    "header_fg": "#f8fafc",
    "header_sub": "#94a3b8",
    "panel_bg": "#ffffff",
    "panel_alt": "#f8fafc",
    "panel_border": "#dbe3ee",
    "text": "#0f172a",
    "muted": "#475569",
    "hint": "#94a3b8",
    "accent": "#2563eb",
    "accent_hover": "#1d4ed8",
    "accent_fg": "#ffffff",
    "question": "#1d4ed8",
    "korean": "#b45309",
    "select_bg": "#dbeafe",
    "select_fg": "#0f172a",
    "row_hover": "#f1f5f9",
    "divider": "#e2e8f0",
    "bar_bg": "#ffffff",
    "bar_fg": "#475569",
    "button_bg": "#f8fafc",
    "button_fg": "#0f172a",
    "button_hover": "#e2e8f0",
    "button_border": "#cbd5e1",
    "header_button_bg": "#1e293b",
    "header_button_hover": "#334155",
    "header_button_fg": "#e2e8f0",
    "input_bg": "#ffffff",
    "input_fg": "#0f172a",
    "note_bg": "#f8fafc",
    "scrollbar_bg": "#cbd5e1",
    "scrollbar_trough": "#ffffff",
    "scrollbar_active": "#94a3b8",
}

DARK_THEME: dict[str, str] = {
    "window": "#0b1220",
    "header_bg": "#111827",
    "header_fg": "#f8fafc",
    "header_sub": "#94a3b8",
    "panel_bg": "#0f172a",
    "panel_alt": "#111c30",
    "panel_border": "#1e293b",
    "text": "#e2e8f0",
    "muted": "#94a3b8",
    "hint": "#64748b",
    "accent": "#60a5fa",
    "accent_hover": "#3b82f6",
    "accent_fg": "#08111f",
    "question": "#93c5fd",
    "korean": "#fbbf24",
    "select_bg": "#1e3a5f",
    "select_fg": "#f8fafc",
    "row_hover": "#16233a",
    "divider": "#1e293b",
    "bar_bg": "#0b1220",
    "bar_fg": "#cbd5e1",
    "button_bg": "#1e293b",
    "button_fg": "#e2e8f0",
    "button_hover": "#334155",
    "button_border": "#334155",
    "header_button_bg": "#1e293b",
    "header_button_hover": "#334155",
    "header_button_fg": "#e2e8f0",
    "input_bg": "#0b1220",
    "input_fg": "#e2e8f0",
    "note_bg": "#0b1220",
    "scrollbar_bg": "#334155",
    "scrollbar_trough": "#0f172a",
    "scrollbar_active": "#475569",
}

THEMES = {"light": LIGHT_THEME, "dark": DARK_THEME}


def _enable_dpi_awareness() -> None:
    """Windows 배율 설정에서도 창 크기가 화면 좌표와 일치하게 합니다."""

    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


class PlaybackError(RuntimeError):
    pass


class SegmentPlayer:
    """ffmpeg로 cue 구간만 뽑아 pygame으로 재생합니다."""

    def __init__(self) -> None:
        try:
            import pygame
        except ImportError as exc:
            raise PlaybackError(
                "pygame이 설치되어 있지 않습니다.\n  python -m pip install pygame"
            ) from exc
        try:
            pygame.mixer.init(frequency=44_100, size=-16, channels=2, buffer=512)
        except Exception as exc:
            raise PlaybackError(f"오디오 장치를 열지 못했습니다: {exc}") from exc
        self._pygame = pygame
        self._sound = None
        self._channel = None

    def play(self, audio_path: Path, start: float, end: float) -> float:
        self.stop()
        duration = max(0.25, float(end) - float(start))
        wav = self._extract(audio_path, start, duration)
        self._sound = self._pygame.mixer.Sound(file=io.BytesIO(wav))
        self._channel = self._sound.play()
        return float(self._sound.get_length())

    def _extract(self, audio_path: Path, start: float, duration: float) -> bytes:
        cmd = [
            find_ffmpeg(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(audio_path),
            "-vn",
            "-f",
            "wav",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise PlaybackError("오디오 구간을 읽는 시간이 초과되었습니다.") from exc
        if proc.returncode != 0 or not proc.stdout:
            detail = proc.stderr.decode("utf-8", errors="replace").strip()
            raise PlaybackError(f"오디오 구간을 읽지 못했습니다.\n{detail[:300]}")
        return proc.stdout

    def stop(self) -> None:
        if self._channel is not None:
            try:
                self._channel.stop()
            except Exception:
                pass
        if self._sound is not None:
            try:
                self._sound.stop()
            except Exception:
                pass
        self._sound = None
        self._channel = None

    def is_busy(self) -> bool:
        return bool(self._channel is not None and self._channel.get_busy())

    def close(self) -> None:
        self.stop()
        try:
            self._pygame.mixer.quit()
        except Exception:
            pass


class CueList(ctk.CTkFrame):
    """문장 번호와 재생 구간만 보여 주는 스크롤 목록.

    영어 본문은 오른쪽 패널에서만 보여 주어 듣기 학습을 방해하지 않습니다.
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        font_head: tuple,
        font_body: tuple,
        colors: dict[str, str],
        on_select: "Callable[[int], None]",
        on_play: "Callable[[], None]",
    ) -> None:
        super().__init__(master, fg_color=colors["panel_bg"], corner_radius=0)
        self._on_select = on_select
        self._on_play = on_play
        self._cues: list = []
        self._ranges: list[tuple[str, str]] = []
        self._current: int | None = None
        self._hovered: int | None = None

        scroll = ctk.CTkScrollbar(
            self,
            width=12,
            fg_color=colors["panel_bg"],
            button_color=colors["scrollbar_bg"],
            button_hover_color=colors["scrollbar_active"],
            command=self._scroll,
        )
        scroll.pack(side="right", fill="y", padx=(0, 6), pady=(2, 8))
        self.text = tk.Text(
            self,
            wrap="none",
            bd=0,
            relief="flat",
            highlightthickness=0,
            padx=14,
            pady=6,
            cursor="arrow",
            takefocus=1,
            font=font_body,
            bg=colors["panel_bg"],
            fg=colors["text"],
            insertbackground=colors["text"],
            selectbackground=colors["select_bg"],
            selectforeground=colors["select_fg"],
            spacing1=7,
            spacing3=7,
            yscrollcommand=scroll.set,
        )
        self.text.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=(0, 4))
        self._scrollbar = scroll

        self.text.tag_configure(
            "num", font=(font_head[0], font_head[1], "bold"), foreground=colors["accent"]
        )
        self.text.tag_configure("time", font=font_head, foreground=colors["muted"])
        self.text.tag_configure("hover", background=colors["row_hover"])
        self.text.tag_configure(
            "current", background=colors["select_bg"], foreground=colors["select_fg"]
        )
        self.text.tag_raise("hover")
        self.text.tag_raise("current")
        self.text.configure(state="disabled")

        self.text.bind("<Button-1>", self._on_click)
        self.text.bind("<Double-Button-1>", self._on_double)
        self.text.bind("<Motion>", self._on_motion)
        self.text.bind("<Leave>", lambda _e: self._clear_hover())
        self.text.bind("<Up>", lambda _e: self._move(-1))
        self.text.bind("<Down>", lambda _e: self._move(1))
        self.text.bind("<Prior>", lambda _e: self._move(-8))
        self.text.bind("<Next>", lambda _e: self._move(8))
        self.text.bind("<Home>", lambda _e: self._jump(0))
        self.text.bind("<End>", lambda _e: self._jump(len(self._cues) - 1))

    def _scroll(self, *args: str) -> None:
        self.text.yview(*args)

    # -- list-like API -----------------------------------------------------

    def _render(self) -> None:
        widget = self.text
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        self._ranges = []
        self._hovered = None
        for i, cue in enumerate(self._cues):
            start = widget.index("end-1c")
            tags = (f"cue{i}",)
            widget.insert("end", f"{cue.index:03d}", ("num",) + tags)
            widget.insert(
                "end",
                f"    {format_mmss(cue.start)}  –  {format_mmss(cue.end)}",
                ("time",) + tags,
            )
            line_end = widget.index("end-1c")
            widget.insert("end", "\n", tags)
            self._ranges.append((start, line_end))
        widget.configure(state="disabled")

    def set_cues(self, cues: list) -> None:
        self._cues = list(cues)
        self._current = None
        self._render()
        self.text.yview_moveto(0)
        self.text.xview_moveto(0)

    def size(self) -> int:
        return len(self._cues)

    def current_index(self) -> int | None:
        return self._current

    def _apply_current(self, index: int, *, see: bool) -> None:
        self._current = index
        widget = self.text
        widget.tag_remove("current", "1.0", "end")
        start, end = self._ranges[index]
        widget.tag_add("current", start, end)
        widget.tag_raise("current")
        if see:
            widget.see(start)
            widget.xview_moveto(0)

    def _clear_hover(self) -> None:
        self._hovered = None
        self.text.tag_remove("hover", "1.0", "end")

    def _on_motion(self, event: tk.Event) -> None:
        index = self._index_at(event.x, event.y)
        if index == self._hovered:
            return
        self._clear_hover()
        if index is None:
            return
        self._hovered = index
        start, end = self._ranges[index]
        self.text.tag_add("hover", start, end)
        self.text.tag_raise("hover")
        self.text.tag_raise("current")

    def select(self, index: int, *, see: bool = True) -> None:
        if not 0 <= index < len(self._cues):
            return
        self._apply_current(index, see=see)
        self._on_select(index)

    def focus_set(self) -> None:  # type: ignore[override]
        self.text.focus_set()

    # -- events ------------------------------------------------------------

    def _index_at(self, x: int, y: int) -> int | None:
        index = self.text.index(f"@{x},{y}")
        for tag in self.text.tag_names(index):
            if tag.startswith("cue") and tag[3:].isdigit():
                return int(tag[3:])
        return None

    def _on_click(self, event: tk.Event) -> str:
        self.text.focus_set()
        index = self._index_at(event.x, event.y)
        if index is not None:
            self.select(index, see=False)
        return "break"

    def _on_double(self, _event: tk.Event) -> str:
        self._on_play()
        return "break"

    def _move(self, delta: int) -> str:
        base = self._current if self._current is not None else 0
        self.select(max(0, min(len(self._cues) - 1, base + delta)))
        return "break"

    def _jump(self, index: int) -> str:
        self.select(index)
        return "break"


class NewProjectDialog(ctk.CTkToplevel):
    """MP3를 선택한 뒤 프로젝트 이름/모델을 정하는 작은 창."""

    def __init__(self, master: tk.Misc, audio_path: Path) -> None:
        super().__init__(master)
        self.result: dict[str, str] | None = None
        self.title("새 프로젝트 만들기")
        self.resizable(False, False)
        self.transient(master)
        colors = getattr(master, "theme", LIGHT_THEME)
        family = tkfont.nametofont("TkDefaultFont").actual("family")
        self.configure(fg_color=colors["panel_bg"])

        body = ctk.CTkFrame(self, fg_color=colors["panel_bg"], corner_radius=0)
        body.pack(fill="both", expand=True, padx=24, pady=20)

        ctk.CTkLabel(
            body,
            text="MP3로 새 프로젝트",
            font=(family, 18, "bold"),
            text_color=colors["text"],
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            body,
            text=audio_path.name,
            font=(family, 11),
            text_color=colors["muted"],
            anchor="w",
        ).pack(fill="x", pady=(2, 16))

        form = ctk.CTkFrame(body, fg_color="transparent")
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        ctk.CTkLabel(
            form,
            text="프로젝트 이름",
            font=(family, 11),
            text_color=colors["text"],
            anchor="w",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.name_var = tk.StringVar(value=audio_path.stem)
        entry = ctk.CTkEntry(
            form,
            textvariable=self.name_var,
            height=36,
            fg_color=colors["input_bg"],
            border_color=colors["button_border"],
            text_color=colors["input_fg"],
            font=(family, 11),
        )
        entry.grid(row=0, column=1, sticky="ew", padx=(14, 0), pady=(0, 8))

        ctk.CTkLabel(
            form,
            text="Whisper 모델",
            font=(family, 11),
            text_color=colors["text"],
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=6)
        self.model_var = tk.StringVar(value="small")
        ctk.CTkOptionMenu(
            form,
            variable=self.model_var,
            values=["base", "small", "medium"],
            width=140,
            height=34,
            fg_color=colors["input_bg"],
            button_color=colors["button_bg"],
            button_hover_color=colors["button_hover"],
            text_color=colors["input_fg"],
            dropdown_fg_color=colors["panel_bg"],
            dropdown_text_color=colors["text"],
            dropdown_hover_color=colors["select_bg"],
            font=(family, 11),
        ).grid(row=1, column=1, sticky="w", padx=(14, 0), pady=6)

        ctk.CTkLabel(
            form,
            text="번역",
            font=(family, 11),
            text_color=colors["text"],
            anchor="w",
        ).grid(row=2, column=0, sticky="w", pady=6)
        ctk.CTkLabel(
            form,
            text="NLLB (오프라인)  ·  Google 번역 사용 안 함",
            font=(family, 11),
            text_color=colors["accent"],
            anchor="w",
        ).grid(row=2, column=1, sticky="w", padx=(14, 0), pady=6)

        note_box = ctk.CTkFrame(
            body, fg_color=colors["note_bg"], corner_radius=12
        )
        note_box.pack(fill="x", pady=(18, 14))
        ctk.CTkLabel(
            note_box,
            text=(
                "NLLB 모델은 최초 1회만 내려받고(약 2.4GB) 이후 로컬 캐시를 재사용합니다.\n"
                "CPU에서 small 모델은 22분 음원 기준 대략 15~20분, 번역에 몇 분 더 걸릴 수 있습니다."
            ),
            font=(family, 10),
            text_color=colors["muted"],
            justify="left",
            anchor="w",
            wraplength=500,
        ).pack(fill="x", padx=14, pady=10)

        buttons = ctk.CTkFrame(body, fg_color="transparent")
        buttons.pack(fill="x")
        ctk.CTkButton(
            buttons,
            text="취소",
            command=self.destroy,
            width=100,
            height=36,
            fg_color=colors["button_bg"],
            hover_color=colors["button_hover"],
            border_width=1,
            border_color=colors["button_border"],
            text_color=colors["button_fg"],
            font=(family, 11),
        ).pack(side="right")
        ctk.CTkButton(
            buttons,
            text="만들기",
            command=self._ok,
            width=110,
            height=36,
            fg_color=colors["accent"],
            hover_color=colors["accent_hover"],
            text_color=colors["accent_fg"],
            font=(family, 11, "bold"),
        ).pack(side="right", padx=(0, 8))

        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.grab_set()
        self.update_idletasks()
        width = max(520, self.winfo_reqwidth())
        height = self.winfo_reqheight()
        x = master.winfo_rootx() + max(0, (master.winfo_width() - width) // 2)
        y = master.winfo_rooty() + max(0, (master.winfo_height() - height) // 3)
        self.geometry(f"{width}x{height}+{x}+{y}")
        entry.focus_set()
        entry.select_range(0, "end")
        self.wait_window()

    def _ok(self) -> None:
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("이름 필요", "프로젝트 이름을 입력하세요.", parent=self)
            return
        self.result = {"name": name, "model": self.model_var.get()}
        self.destroy()


class ProgressDialog(ctk.CTkToplevel):
    """프로젝트 생성 진행률을 보여주는 창."""

    def __init__(self, master: tk.Misc, *, note: str) -> None:
        super().__init__(master)
        self.title("프로젝트 생성 중")
        self.resizable(False, False)
        self.transient(master)
        colors = getattr(master, "theme", LIGHT_THEME)
        family = tkfont.nametofont("TkDefaultFont").actual("family")
        self.configure(fg_color=colors["panel_bg"])
        self.queue: "queue.Queue[tuple[str, tuple]]" = queue.Queue()
        self.cancel_event = threading.Event()
        self.started = time.time()
        self._closed = False
        self._bar_running = False

        body = ctk.CTkFrame(self, fg_color=colors["panel_bg"], corner_radius=0)
        body.pack(fill="both", expand=True, padx=26, pady=22)
        ctk.CTkLabel(
            body,
            text="프로젝트를 만들고 있습니다",
            font=(family, 18, "bold"),
            text_color=colors["text"],
            anchor="w",
        ).pack(fill="x")
        self.note_var = tk.StringVar(value=note)
        ctk.CTkLabel(
            body,
            textvariable=self.note_var,
            font=(family, 10),
            text_color=colors["muted"],
            anchor="w",
            justify="left",
            wraplength=500,
        ).pack(fill="x", pady=(4, 16))

        self.status_var = tk.StringVar(value="준비 중…")
        ctk.CTkLabel(
            body,
            textvariable=self.status_var,
            font=(family, 11),
            text_color=colors["text"],
            anchor="w",
        ).pack(fill="x")
        self.bar = ctk.CTkProgressBar(
            body,
            height=10,
            corner_radius=5,
            fg_color=colors["input_bg"],
            progress_color=colors["accent"],
        )
        self.bar.set(0)
        self.bar.pack(fill="x", pady=(10, 10))
        self.elapsed_var = tk.StringVar(value="경과 00:00")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(
            row,
            textvariable=self.elapsed_var,
            font=(family, 10),
            text_color=colors["hint"],
            anchor="w",
        ).pack(side="left")
        self.cancel_btn = ctk.CTkButton(
            row,
            text="취소",
            command=self.request_cancel,
            width=100,
            height=34,
            fg_color=colors["button_bg"],
            hover_color=colors["button_hover"],
            border_width=1,
            border_color=colors["button_border"],
            text_color=colors["button_fg"],
            font=(family, 11),
        )
        self.cancel_btn.pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self.request_cancel)
        self.grab_set()
        self.update_idletasks()
        width = max(560, self.winfo_reqwidth())
        height = self.winfo_reqheight()
        x = master.winfo_rootx() + max(0, (master.winfo_width() - width) // 2)
        y = master.winfo_rooty() + max(0, (master.winfo_height() - height) // 3)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.after(300, self._tick)

    def push(self, kind: str, payload: tuple) -> None:
        self.queue.put((kind, payload))

    def set_note(self, text: str) -> None:
        self.note_var.set(text)

    def set_progress(self, message: str, fraction: float | None) -> None:
        self.status_var.set(message)
        if fraction is None:
            if not self._bar_running:
                self.bar.configure(mode="indeterminate")
                self.bar.start()
                self._bar_running = True
            return
        if self._bar_running:
            self.bar.stop()
            self.bar.configure(mode="determinate")
            self._bar_running = False
        self.bar.set(max(0.0, min(1.0, fraction)))

    def request_cancel(self) -> None:
        if self.cancel_event.is_set():
            return
        self.cancel_event.set()
        self.cancel_btn.configure(state="disabled")
        self.status_var.set("취소 요청됨 — 현재 단계가 끝나면 중단됩니다.")

    def _tick(self) -> None:
        if self._closed:
            return
        elapsed = int(time.time() - self.started)
        self.elapsed_var.set(f"경과 {elapsed // 60:02d}:{elapsed % 60:02d}")
        self.after(500, self._tick)

    def close(self) -> None:
        self._closed = True
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()


class TailFixReport(ctk.CTkToplevel):
    """끝 잘림 보정 결과와 바뀐 문장 번호를 보여주는 창."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        fixes: list[TailFix],
        saved_paths: list[Path],
    ) -> None:
        super().__init__(master)
        colors = getattr(master, "theme", LIGHT_THEME)
        self.title("끝 잘림 보정 결과")
        self.transient(master)
        self.configure(fg_color=colors["panel_bg"])
        self.resizable(True, True)

        body = ctk.CTkFrame(self, fg_color=colors["panel_bg"], corner_radius=0)
        body.pack(fill="both", expand=True, padx=24, pady=20)

        family = tkfont.nametofont("TkDefaultFont").actual("family")
        ctk.CTkLabel(
            body,
            text=f"끝이 잘린 문장 {len(fixes)}개를 보정했습니다",
            font=(family, 16, "bold"),
            text_color=colors["text"],
            anchor="w",
        ).pack(fill="x")

        total = sum(fix.extension for fix in fixes)
        shifted = sum(1 for fix in fixes if fix.shifts_next)
        saved = saved_paths[0].parent if saved_paths else None
        note = f"종료 시간을 모두 {total:.2f}초 늘렸습니다."
        if shifted:
            note += (
                f"\n다음 문장과 경계가 겹치던 {shifted}곳은 "
                "다음 문장 시작도 함께 뒤로 옮겼습니다."
            )
        if saved is not None:
            note += f"\n저장 위치: {saved}"
        ctk.CTkLabel(
            body,
            text=note,
            font=(family, 10),
            text_color=colors["muted"],
            anchor="w",
            justify="left",
            wraplength=640,
        ).pack(fill="x", pady=(4, 14))

        mono = getattr(master, "font_list", (family, 10))
        text = ctk.CTkTextbox(
            body,
            wrap="none",
            corner_radius=12,
            border_width=1,
            border_color=colors["panel_border"],
            fg_color=colors["input_bg"],
            text_color=colors["text"],
            font=mono,
        )
        text.pack(fill="both", expand=True)

        lines: list[str] = []
        for fix in fixes:
            lines.append(
                f"{fix.index:03d}   {_format_seconds(fix.old_end)} → "
                f"{_format_seconds(fix.new_end)}   (+{fix.extension:.2f}초)"
            )
            if (
                fix.shifts_next
                and fix.next_old_start is not None
                and fix.next_new_start is not None
            ):
                lines.append(
                    f"      ↳ {fix.next_index:03d} 시작 "
                    f"{_format_seconds(fix.next_old_start)} → "
                    f"{_format_seconds(fix.next_new_start)}"
                )
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

        buttons = ctk.CTkFrame(body, fg_color="transparent")
        buttons.pack(fill="x", pady=(14, 0))
        ctk.CTkButton(
            buttons,
            text="닫기",
            command=self.destroy,
            width=100,
            height=36,
            fg_color=colors["button_bg"],
            hover_color=colors["button_hover"],
            border_width=1,
            border_color=colors["button_border"],
            text_color=colors["button_fg"],
            font=(family, 11),
        ).pack(side="right")

        self.bind("<Escape>", lambda _event: self.destroy())
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.update_idletasks()
        width = max(520, min(720, self.winfo_reqwidth()))
        height = max(380, min(620, self.winfo_reqheight() + 80))
        x = master.winfo_rootx() + max(0, (master.winfo_width() - width) // 2)
        y = master.winfo_rooty() + max(0, (master.winfo_height() - height) // 3)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.grab_set()
        self.focus_set()


class CueEditDialog(ctk.CTkToplevel):
    """선택한 문장의 영어 지문과 한국어 번역을 함께 수정합니다."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        english: str,
        korean: str,
        colors: dict[str, str],
        font_english: tuple,
        font_korean: tuple,
        font_info: tuple,
        font_small: tuple,
    ) -> None:
        super().__init__(master)
        self.result: tuple[str, str] | None = None
        self._colors = colors
        self.title("문장 수정")
        self.transient(master)
        self.configure(fg_color=colors["panel_bg"])
        self.resizable(True, True)

        body = ctk.CTkFrame(self, fg_color=colors["panel_bg"], corner_radius=0)
        body.pack(fill="both", expand=True, padx=22, pady=18)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        body.rowconfigure(3, weight=1)

        self.english = self._add_field(
            body,
            "영어 지문",
            english,
            font=font_english,
            colors=colors,
            font_info=font_info,
            row=0,
        )
        self.korean = self._add_field(
            body,
            "한국어 번역",
            korean,
            font=font_korean,
            colors=colors,
            font_info=font_info,
            row=2,
        )

        footer = ctk.CTkFrame(body, fg_color="transparent")
        footer.grid(row=4, column=0, sticky="ew")
        ctk.CTkLabel(
            footer,
            text="Ctrl+Enter: 저장   ·   Esc: 취소",
            font=font_small,
            text_color=colors["hint"],
            anchor="w",
        ).pack(side="left")
        ctk.CTkButton(
            footer,
            text="취소",
            command=self._cancel,
            width=100,
            height=36,
            fg_color=colors["button_bg"],
            hover_color=colors["button_hover"],
            border_width=1,
            border_color=colors["button_border"],
            text_color=colors["button_fg"],
            font=font_small,
        ).pack(side="right")
        ctk.CTkButton(
            footer,
            text="저장",
            command=self._save,
            width=100,
            height=36,
            fg_color=colors["accent"],
            hover_color=colors["accent_hover"],
            text_color=colors["accent_fg"],
            font=font_small,
        ).pack(side="right", padx=(0, 8))

        self.bind("<Escape>", lambda _event: self._cancel())
        self.bind("<Control-Return>", lambda _event: self._save())
        self.protocol("WM_DELETE_WINDOW", self._cancel)

        self.update_idletasks()
        width = max(620, min(900, (master.winfo_width() or 720) - 160))
        height = max(520, min(720, (master.winfo_height() or 620) - 80))
        self.minsize(620, 520)
        x = master.winfo_rootx() + max(0, (master.winfo_width() - width) // 2)
        y = master.winfo_rooty() + max(0, (master.winfo_height() - height) // 3)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.grab_set()
        self.english.focus_set()

    def _add_field(
        self,
        parent: tk.Misc,
        title: str,
        value: str,
        *,
        font: tuple,
        colors: dict[str, str],
        font_info: tuple,
        row: int,
    ) -> ctk.CTkTextbox:
        ctk.CTkLabel(
            parent,
            text=title,
            font=font_info,
            text_color=colors["text"],
            anchor="w",
        ).grid(row=row, column=0, sticky="ew")

        frame = ctk.CTkFrame(
            parent,
            fg_color=colors["input_bg"],
            corner_radius=12,
            border_width=1,
            border_color=colors["panel_border"],
        )
        frame.grid(row=row + 1, column=0, sticky="nsew", pady=(6, 14))

        text = ctk.CTkTextbox(
            frame,
            wrap="word",
            height=120,
            corner_radius=10,
            border_width=0,
            fg_color=colors["input_bg"],
            text_color=colors["input_fg"],
            scrollbar_button_color=colors["scrollbar_bg"],
            scrollbar_button_hover_color=colors["scrollbar_active"],
            font=font,
        )
        text.pack(fill="both", expand=True, padx=8, pady=8)
        text._textbox.configure(
            padx=4,
            pady=4,
            insertbackground=colors["input_fg"],
            selectbackground=colors["select_bg"],
            selectforeground=colors["select_fg"],
        )
        text.insert("1.0", value)
        text.mark_set("insert", "end-1c")
        return text

    def _save(self) -> None:
        english = self.english.get("1.0", "end-1c").strip()
        if not english:
            messagebox.showwarning(
                "내용 없음", "영어 지문을 입력하세요.", parent=self
            )
            self.english.focus_set()
            return
        korean = self.korean.get("1.0", "end-1c").strip()
        self.result = (english, korean)
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.destroy()


def _format_seconds(seconds: float) -> str:
    total = max(0.0, float(seconds))
    minutes, secs = divmod(total, 60.0)
    return f"{int(minutes):02d}:{secs:05.2f}"


class StudyApp(ctk.CTk):
    LEFT_WIDTH = 500
    ENGLISH_MIN_LINES = 2
    ENGLISH_MAX_LINES = 6
    KOREAN_MIN_LINES = 1
    KOREAN_MAX_LINES = 8

    def __init__(
        self,
        project: Project | None,
        projects: list[Project] | None = None,
        *,
        theme: str = "light",
    ) -> None:
        _enable_dpi_awareness()
        super().__init__()
        self.project = project
        if projects is not None:
            self.projects = projects
        else:
            self.projects = [project] if project is not None else []
        self.theme_name = theme if theme in THEMES else "light"
        self.theme = THEMES[self.theme_name]
        ctk.set_appearance_mode(self.theme_name)
        self.show_korean = True
        self.playing_index: int | None = None
        self._stop_after: str | None = None
        self.player: SegmentPlayer | None = None
        self._player_error: str | None = None
        self._tail_busy = False
        self._tail_queue: "queue.Queue[tuple[str, object]] | None" = None
        self._tail_project_dir: Path | None = None
        self._state_after: str | None = None
        self._theme_after: str | None = None
        self._fitting = False

        self.title(f"{project.name} — 영어 듣기 학습" if project else "영어 듣기 학습")
        self.geometry(self._center_geometry(1180, 760))
        self.minsize(900, 560)
        self.configure(fg_color=self.theme["window"])

        self._init_fonts()
        self._init_player()
        self._build_ui()
        self._bind_keys()
        self._populate(self._saved_index(self.project))
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.cue_list.focus_set()

    # ---------------------------------------------------------------- setup

    def _center_geometry(self, width: int, height: int) -> str:
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 3)
        return f"{width}x{height}+{x}+{y}"

    def _project_subtitle(self, project: Project) -> str:
        created = project.created_at[:10] if project.created_at else ""
        subtitle = f"{project.cue_count}개 문장"
        if created:
            subtitle += f"  ·  생성 {created}"
        return subtitle

    def _init_fonts(self) -> None:
        families = set(tkfont.families(self))
        korean = next(
            (f for f in ("Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "NanumGothic") if f in families),
            tkfont.nametofont("TkDefaultFont").actual("family"),
        )
        body = next(
            (f for f in ("Segoe UI", "Helvetica Neue", "Arial") if f in families),
            korean,
        )
        mono = next(
            (f for f in ("Consolas", "Menlo", "DejaVu Sans Mono", "Courier New") if f in families),
            body,
        )
        self.font_title = (korean, 20, "bold")
        self.font_sub = (korean, 10)
        self.font_list = (mono, 10)
        self.font_card = (korean, 12)
        self.font_english = (body, 21)
        self.font_korean = (korean, 15)
        self.font_small = (korean, 10)
        self.font_info = (korean, 11, "bold")

    def _init_player(self) -> None:
        try:
            self.player = SegmentPlayer()
        except PlaybackError as exc:
            self._player_error = str(exc)
            self.player = None

    def _theme_button_text(self) -> str:
        return "라이트" if self.theme_name == "dark" else "다크"

    def _make_header_button(
        self,
        parent: tk.Misc,
        text: str,
        command: Callable[[], object],
        *,
        width: int = 120,
    ) -> ctk.CTkButton:
        c = self.theme
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=width,
            height=36,
            corner_radius=10,
            fg_color=c["header_button_bg"],
            hover_color=c["header_button_hover"],
            border_width=1,
            border_color=c["header_button_hover"],
            text_color=c["header_button_fg"],
            font=(self.font_small[0], 10, "bold"),
        )

    def _make_bar_button(
        self,
        parent: tk.Misc,
        text: str,
        command: Callable[[], object],
        *,
        width: int = 110,
    ) -> ctk.CTkButton:
        c = self.theme
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            width=width,
            height=36,
            corner_radius=10,
            fg_color=c["button_bg"],
            hover_color=c["button_hover"],
            border_width=1,
            border_color=c["button_border"],
            text_color=c["button_fg"],
            font=(self.font_small[0], 10),
        )

    def _build_ui(self) -> None:
        c = self.theme
        family = self.font_small[0]

        header = ctk.CTkFrame(self, fg_color=c["header_bg"], corner_radius=0)
        header.pack(side="top", fill="x")

        # ?? ??? ?? ?????. ???? ??? ??? ?? ???
        # ??? ??? ??? ??? ??? ?? ?? ?????.
        actions = ctk.CTkFrame(header, fg_color="transparent")
        actions.pack(side="right", padx=(0, 16), pady=14)
        self.new_btn = self._make_header_button(
            actions,
            "+ MP3? ? ????",
            self._new_project,
            width=174,
        )
        self.new_btn.pack(side="right")
        self._make_header_button(
            actions, "???? ??", self._choose_project, width=124
        ).pack(side="right", padx=(8, 0))
        self._make_header_button(
            actions, self._theme_button_text(), self._toggle_theme, width=76
        ).pack(side="right", padx=(8, 0))

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.pack(side="left", fill="x", expand=True, padx=(24, 12), pady=14)
        self.header_name = tk.StringVar(
            value=self.project.name if self.project else "? ????"
        )
        ctk.CTkLabel(
            title_box,
            textvariable=self.header_name,
            font=self.font_title,
            text_color=c["header_fg"],
            anchor="w",
        ).pack(fill="x")
        self.header_sub = tk.StringVar(
            value=(
                self._project_subtitle(self.project)
                if self.project
                else "??? ? [+ MP3? ? ????]? ?? ?????"
            )
        )
        ctk.CTkLabel(
            title_box,
            textvariable=self.header_sub,
            font=self.font_sub,
            text_color=c["header_sub"],
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

        # ?? ?? body?? ?? pack?? ? ??? ??? ?? ????.
        bar = ctk.CTkFrame(self, fg_color=c["bar_bg"], corner_radius=0)
        bar.pack(side="bottom", fill="x")
        ctk.CTkFrame(
            self, fg_color=c["panel_border"], height=1, corner_radius=0
        ).pack(side="bottom", fill="x")
        bar_inner = ctk.CTkFrame(bar, fg_color="transparent")
        bar_inner.pack(fill="x", padx=16, pady=8)

        self.play_btn = ctk.CTkButton(
            bar_inner,
            text="?  ?? (Space)",
            command=self._play_selected,
            width=154,
            height=36,
            corner_radius=10,
            fg_color=c["accent"],
            hover_color=c["accent_hover"],
            text_color=c["accent_fg"],
            font=(family, 10, "bold"),
        )
        self.play_btn.pack(side="left")
        self._make_bar_button(
            bar_inner, "?  ?? (Esc)", self._stop_playback, width=128
        ).pack(side="left", padx=(8, 0))
        self._make_bar_button(
            bar_inner, "?? ??", lambda: self._step(-1), width=100
        ).pack(side="left", padx=(18, 0))
        self._make_bar_button(
            bar_inner, "?? ??", lambda: self._step(1), width=100
        ).pack(side="left", padx=8)
        self.tail_btn = self._make_bar_button(
            bar_inner, "? ?? ??", self._fix_tail_clipping, width=120
        )
        self.tail_btn.pack(side="left", padx=(18, 0))
        if self._tail_busy:
            self.tail_btn.configure(state="disabled")

        self.mode_var = tk.StringVar(
            value="??: ??" if self.show_korean else "??: ???"
        )
        ctk.CTkLabel(
            bar_inner,
            textvariable=self.mode_var,
            font=self.font_small,
            fg_color=c["panel_alt"],
            corner_radius=8,
            height=28,
            width=96,
            text_color=c["bar_fg"],
        ).pack(side="left", padx=18)

        self.status_var = tk.StringVar(value="???")
        ctk.CTkLabel(
            bar_inner,
            textvariable=self.status_var,
            font=self.font_small,
            text_color=c["bar_fg"],
            anchor="e",
        ).pack(side="right", padx=(12, 0))

        if self._player_error:
            self.status_var.set("??? ?? ??")
            self.play_btn.configure(state="disabled")

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(side="top", fill="both", expand=True, padx=16, pady=(14, 12))

        paned = tk.PanedWindow(
            body,
            orient="horizontal",
            sashwidth=12,
            bg=c["window"],
            bd=0,
            relief="flat",
            sashrelief="flat",
        )
        paned.pack(fill="both", expand=True)

        left = ctk.CTkFrame(
            paned,
            fg_color=c["panel_bg"],
            bg_color=c["window"],
            corner_radius=16,
            border_width=1,
            border_color=c["panel_border"],
        )
        right = ctk.CTkFrame(
            paned,
            fg_color=c["panel_bg"],
            bg_color=c["window"],
            corner_radius=16,
            border_width=1,
            border_color=c["panel_border"],
        )
        paned.add(left, minsize=340, width=self.LEFT_WIDTH)
        paned.add(right, minsize=420, width=680)

        # left: sentence list
        list_head = ctk.CTkFrame(left, fg_color="transparent")
        list_head.pack(fill="x", padx=18, pady=(16, 8))
        ctk.CTkLabel(
            list_head,
            text="?? ??",
            font=(family, 13, "bold"),
            text_color=c["text"],
            anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            list_head,
            text="SPACE  ??",
            font=(family, 9, "bold"),
            fg_color=c["panel_alt"],
            corner_radius=7,
            height=24,
            width=86,
            text_color=c["hint"],
        ).pack(side="right")
        ctk.CTkFrame(
            left, fg_color=c["divider"], height=1, corner_radius=0
        ).pack(fill="x", padx=18)

        self.cue_list = CueList(
            left,
            font_head=(self.font_list[0], 9),
            font_body=self.font_card,
            colors=c,
            on_select=self._on_select,
            on_play=self._play_selected,
        )
        self.cue_list.pack(fill="both", expand=True, padx=(8, 8), pady=(6, 12))

        # right: study view
        view = ctk.CTkFrame(right, fg_color="transparent")
        view.pack(fill="both", expand=True, padx=30, pady=(22, 18))

        self.info_var = tk.StringVar(value="??? ?????")
        info_row = ctk.CTkFrame(view, fg_color="transparent")
        info_row.pack(fill="x")
        self.edit_btn = ctk.CTkButton(
            info_row,
            text="??",
            command=self._edit_cue,
            width=72,
            height=32,
            corner_radius=9,
            fg_color=c["button_bg"],
            hover_color=c["button_hover"],
            border_width=1,
            border_color=c["button_border"],
            text_color=c["button_fg"],
            font=(family, 10, "bold"),
        )
        self.edit_btn.pack(side="right")
        ctk.CTkLabel(
            info_row,
            textvariable=self.info_var,
            font=self.font_info,
            text_color=c["muted"],
            anchor="w",
        ).pack(side="left", fill="x", expand=True, padx=(0, 10))

        ctk.CTkFrame(
            view, fg_color=c["divider"], height=1, corner_radius=0
        ).pack(fill="x", pady=(12, 20))

        self.english_text = self._make_display_text(
            view, font=self.font_english, fg=c["text"]
        )
        self.english_text.pack(fill="x", anchor="w")

        self.korean_text = self._make_display_text(
            view, font=self.font_korean, fg=c["muted"]
        )
        self.korean_text.pack(fill="x", anchor="w", pady=(18, 0))

        self.hint_label = ctk.CTkLabel(
            view,
            text=(
                "Space: ??/??   ?   Enter: ?? ??\n"
                "j/k: ?/?? ?? + ??   ?   Esc: ??   ?   Ctrl+D: ??"
            ),
            font=self.font_small,
            text_color=c["hint"],
            anchor="w",
            justify="left",
        )
        self.hint_label.pack(fill="x", side="bottom", pady=(18, 0))
        ctk.CTkFrame(view, fg_color="transparent").pack(
            fill="both", expand=True
        )

        view.bind("<Configure>", self._on_view_resize)

    @staticmethod
    def _text_inner(widget: tk.Misc) -> tk.Text:
        """CTkTextbox ??? ?? tk.Text? ?????."""

        return getattr(widget, "_textbox", widget)

    def _display_line_height(self, widget: tk.Misc) -> int:
        inner = self._text_inner(widget)
        try:
            font = tkfont.Font(font=inner.cget("font"))
            return max(1, int(font.metrics("linespace")))
        except (tk.TclError, TypeError, ValueError):
            return 24

    def _make_display_text(
        self, parent: tk.Misc, *, font: tuple, fg: str
    ) -> ctk.CTkTextbox:
        """??/??? ??? ?? ??? ?? ?? ?? ?? ??."""

        c = self.theme
        widget = ctk.CTkTextbox(
            parent,
            wrap="word",
            height=self.ENGLISH_MIN_LINES * 30,
            corner_radius=0,
            border_width=0,
            border_spacing=0,
            fg_color=c["panel_bg"],
            text_color=fg,
            font=font,
            activate_scrollbars=False,
        )
        inner = widget._textbox
        inner.configure(
            padx=0,
            pady=0,
            cursor="xterm",
            selectbackground=c["select_bg"],
            selectforeground=c["select_fg"],
        )
        widget.configure(state="disabled")
        self._bind_copy_menu(widget)
        return widget

    def _bind_copy_menu(self, widget: ctk.CTkTextbox) -> None:
        inner = self._text_inner(widget)
        menu = tk.Menu(
            inner,
            tearoff=0,
            bg=self.theme["panel_bg"],
            fg=self.theme["text"],
            activebackground=self.theme["select_bg"],
            activeforeground=self.theme["select_fg"],
            bd=0,
        )
        menu.add_command(
            label="??", command=lambda: inner.event_generate("<<Copy>>")
        )
        menu.add_command(
            label="?? ??",
            command=lambda: inner.tag_add("sel", "1.0", "end-1c"),
        )
        inner.bind(
            "<Button-3>",
            lambda event: menu.tk_popup(event.x_root, event.y_root),
        )

    def _set_display_text(
        self,
        widget: ctk.CTkTextbox,
        value: str,
        *,
        min_lines: int,
        max_lines: int,
        fg: str,
    ) -> None:
        widget.configure(state="normal", text_color=fg)
        widget.delete("1.0", "end")
        if value:
            # Text는 끝에 암묵적인 개행을 갖기 때문에, 실제 개행이 있으면
            # 마지막 빈 줄까지 한 줄로 세어 높이가 커집니다.
            widget.insert("1.0", value.rstrip("\n"))
        widget.configure(state="disabled")
        self._fit_display_text(widget, min_lines=min_lines, max_lines=max_lines)

    def _fit_display_text(
        self, widget: ctk.CTkTextbox, *, min_lines: int, max_lines: int
    ) -> None:
        if self._fitting:
            return
        line_height = self._display_line_height(widget)
        if not widget.winfo_ismapped():
            widget.configure(height=min_lines * line_height + 4)
            return
        self._fitting = True
        try:
            inner = self._text_inner(widget)
            inner.update_idletasks()
            try:
                # 'end-1c'는 마지막 줄(부분 줄)을 빼고 세므로, 줄바꿈으로
                # 여러 줄이 된 본문의 높이가 1줄로 줄어드는 문제가 있습니다.
                counts = inner.count("1.0", "end", "displaylines")
                lines = int(counts[0]) if counts else min_lines
            except (tk.TclError, TypeError, IndexError):
                lines = min_lines
            lines = max(min_lines, min(max_lines, lines))
            widget.configure(height=lines * line_height + 4)
        finally:
            self._fitting = False

    def _toggle_theme(self) -> str:
        index = self._current_index()
        status = self.status_var.get() if hasattr(self, "status_var") else ""
        self.theme_name = "dark" if self.theme_name == "light" else "light"
        self.theme = THEMES[self.theme_name]
        ctk.set_appearance_mode(self.theme_name)
        self.configure(fg_color=self.theme["window"])
        # CustomTkinter는 테마를 바꿀 때 제목 표시줄 색을 갱신하면서
        # `after(1, 포커스_복원)`을 예약합니다. 그 직후에 위젯을 모두
        # 파괴하면 이미 사라진 위젯에 포커스를 주려다 TclError가 납니다.
        # 예약된 콜백이 먼저 실행되도록 재구성을 잠시 미룹니다.
        if self._theme_after is not None:
            try:
                self.after_cancel(self._theme_after)
            except Exception:
                pass
        self._theme_after = self.after(
            30, lambda: self._rebuild_theme(index, status)
        )
        return "break"

    def _rebuild_theme(self, index: int | None, status: str) -> None:
        self._theme_after = None
        for child in self.winfo_children():
            child.destroy()
        self._build_ui()
        self._populate(index)
        if status:
            self.status_var.set(status)
        self.cue_list.focus_set()

    def _bind_keys(self) -> None:
        self.bind("<space>", self._on_space)
        self.bind("<Return>", self._on_enter)
        self.bind("<KP_Enter>", self._on_enter)
        self.bind("<Escape>", lambda _e: self._stop_playback() or "break")
        # j = 위, k = 아래 (선택 후 바로 재생)
        self.bind("<j>", lambda _e: self._move_and_play(1))
        self.bind("<k>", lambda _e: self._move_and_play(-1))
        self.bind("<J>", lambda _e: self._move_and_play(1))
        self.bind("<K>", lambda _e: self._move_and_play(-1))
        self.bind("<Control-o>", lambda _e: self._choose_project())
        self.bind("<Control-d>", lambda _e: self._toggle_theme())
        self.bind("<F5>", lambda _e: self._reload())

    # ------------------------------------------------------------- contents

    def _populate(self, index: int | None = None) -> None:
        cues = self.project.cues if self.project else []
        self.cue_list.set_cues(cues)
        self.title(f"{self.project.name} — 영어 듣기 학습" if self.project else "영어 듣기 학습")
        if cues:
            target = index if index is not None else 0
            self.cue_list.select(max(0, min(len(cues) - 1, target)))
        self._update_view()

    def _current_index(self) -> int | None:
        return self.cue_list.current_index()

    @staticmethod
    def _saved_index(project: Project | None) -> int | None:
        if project is None:
            return None
        return load_last_index(project.directory)

    def _schedule_save_position(self) -> None:
        """선택이 바뀔 때마다 쓰지 않고 잠시 모았다가 저장합니다."""

        if self.project is None:
            return
        if self._state_after is not None:
            try:
                self.after_cancel(self._state_after)
            except Exception:
                pass
        self._state_after = self.after(400, self._save_position)

    def _save_position(self) -> None:
        self._state_after = None
        if self.project is None:
            return
        index = self._current_index()
        if index is not None:
            save_last_index(self.project.directory, index)

    def _flush_position(self) -> None:
        """대기 중인 저장을 즉시 반영합니다."""

        if self._state_after is not None:
            try:
                self.after_cancel(self._state_after)
            except Exception:
                pass
            self._state_after = None
        self._save_position()

    def _update_view(self) -> None:
        c = self.theme
        idx = self._current_index()
        if idx is None or self.project is None:
            self.info_var.set(
                "문장을 선택하세요" if self.project else "먼저 프로젝트를 만드세요"
            )
            self.edit_btn.configure(state="disabled")
            self._set_display_text(
                self.english_text,
                "",
                min_lines=self.ENGLISH_MIN_LINES,
                max_lines=self.ENGLISH_MAX_LINES,
                fg=c["text"],
            )
            self._set_display_text(
                self.korean_text,
                (
                    "오른쪽 위 [+ MP3로 새 프로젝트]를 눌러 MP3를 고르면\n"
                    "자막을 만들고 바로 공부할 수 있습니다."
                    if self.project is None
                    else ""
                ),
                min_lines=self.KOREAN_MIN_LINES,
                max_lines=self.KOREAN_MAX_LINES,
                fg=c["hint"],
            )
            return

        cue = self.project.cues[idx]
        question = f"  ·  {cue.question}번 문제" if cue.question else ""
        self.info_var.set(
            f"문장 {cue.index:03d} / {len(self.project.cues):03d}"
            f"   ·   {format_mmss(cue.start)}–{format_mmss(cue.end)}{question}"
        )
        self.edit_btn.configure(state="normal")

        if cue.lang == "ko":
            self._set_display_text(
                self.english_text,
                cue.text,
                min_lines=self.ENGLISH_MIN_LINES,
                max_lines=self.ENGLISH_MAX_LINES,
                fg=c["korean"],
            )
            self._set_display_text(
                self.korean_text,
                "한국어 안내 문장입니다.",
                min_lines=self.KOREAN_MIN_LINES,
                max_lines=self.KOREAN_MAX_LINES,
                fg=c["hint"],
            )
            return

        self._set_display_text(
            self.english_text,
            cue.text,
            min_lines=self.ENGLISH_MIN_LINES,
            max_lines=self.ENGLISH_MAX_LINES,
            fg=c["text"],
        )
        if self.show_korean:
            value = cue.translation or "(번역 없음)"
            fg = c["muted"]
        else:
            value = "한국어 번역 숨김  ·  Enter로 다시 표시"
            fg = c["hint"]
        self._set_display_text(
            self.korean_text,
            value,
            min_lines=self.KOREAN_MIN_LINES,
            max_lines=self.KOREAN_MAX_LINES,
            fg=fg,
        )

    def _on_view_resize(self, _event: tk.Event) -> None:
        self._fit_display_text(
            self.english_text,
            min_lines=self.ENGLISH_MIN_LINES,
            max_lines=self.ENGLISH_MAX_LINES,
        )
        self._fit_display_text(
            self.korean_text,
            min_lines=self.KOREAN_MIN_LINES,
            max_lines=self.KOREAN_MAX_LINES,
        )

    def _on_select(self, _index: int) -> None:
        self._update_view()
        self._schedule_save_position()

    def _edit_cue(self) -> None:
        if self.project is None:
            return
        idx = self._current_index()
        if idx is None:
            return
        cue = self.project.cues[idx]

        dialog = CueEditDialog(
            self,
            english=cue.text,
            korean=cue.translation,
            colors=self.theme,
            font_english=self.font_english,
            font_korean=self.font_korean,
            font_info=self.font_info,
            font_small=self.font_small,
        )
        self.wait_window(dialog)
        if dialog.result is None:
            return

        new_text, new_translation = dialog.result
        if (new_text, new_translation) == (cue.text, cue.translation):
            return

        old_text, old_translation = cue.text, cue.translation
        cue.text = new_text
        cue.translation = new_translation
        try:
            save_project_cues(self.project)
        except Exception as exc:  # noqa: BLE001 - 저장 실패를 사용자에게 알림
            cue.text = old_text
            cue.translation = old_translation
            messagebox.showerror(
                "저장 실패",
                f"문장을 저장하지 못했습니다.\n{exc}",
            )
            return

        self._update_view()
        self.status_var.set(f"문장 {cue.index:03d} 수정 저장 완료")

    # ------------------------------------------------------------- playback

    def _play_selected(self) -> None:
        idx = self._current_index()
        if idx is None or self.project is None:
            return
        cue = self.project.cues[idx]
        if self.player is None:
            messagebox.showerror("재생할 수 없음", self._player_error or "오디오 장치 오류")
            return
        if self.playing_index == cue.index and self.player.is_busy():
            self._stop_playback()
            return
        self._stop_playback(quiet=True)
        self.status_var.set(f"▶ {cue.index:03d} 준비 중…")
        self.update_idletasks()
        # 단어 타임스탬프는 마지막 자음/여운보다 조금 일찍 끝나므로
        # 앞뒤로 여유를 주되, 다음 문장의 시작은 침범하지 않습니다.
        start = max(0.0, cue.start - 0.08)
        end = cue.end + 0.45
        if idx + 1 < len(self.project.cues):
            next_start = self.project.cues[idx + 1].start
            if next_start > cue.end + 1e-6:
                end = min(end, max(cue.start + 0.25, next_start - 0.03))
            else:
                # 다음 문장과 경계가 정확히 겹치면 저장된 종료보다 먼저
                # 재생을 끊지 않습니다. 꼬리 잘림 보정 전에도 들을 수 있게
                # 짧은 여유를 남깁니다.
                end = cue.end + 0.30
        elif self.project.duration:
            end = min(end, self.project.duration)
        try:
            length = self.player.play(self.project.audio, start, end)
        except Exception as exc:
            self.status_var.set("재생 실패")
            messagebox.showerror("재생 실패", str(exc))
            return
        self.playing_index = cue.index
        self.status_var.set(
            f"▶ {cue.index:03d}  {format_mmss(cue.start)}–{format_mmss(cue.end)} 재생 중"
        )
        self._stop_after = self.after(int(length * 1000) + 40, self._playback_finished)

    def _playback_finished(self) -> None:
        self._stop_after = None
        self.playing_index = None
        self.status_var.set("재생 완료")

    def _stop_playback(self, quiet: bool = False) -> None:
        if self._stop_after is not None:
            try:
                self.after_cancel(self._stop_after)
            except Exception:
                pass
            self._stop_after = None
        if self.player is not None:
            self.player.stop()
        had = self.playing_index is not None
        self.playing_index = None
        if had and not quiet:
            self.status_var.set("■ 정지")

    def _on_space(self, _event: tk.Event | None = None) -> str:
        self._play_selected()
        return "break"

    def _on_enter(self, _event: tk.Event | None = None) -> str:
        self.show_korean = not self.show_korean
        self.mode_var.set("표시: 한영" if self.show_korean else "표시: 영어만")
        self._update_view()
        return "break"

    def _step(self, delta: int) -> None:
        idx = self._current_index()
        if idx is None:
            return
        nxt = max(0, min(len(self.project.cues) - 1, idx + delta))
        self.cue_list.select(nxt)
        self.cue_list.focus_set()

    def _move_and_play(self, delta: int) -> str:
        if self.project is None or not self.project.cues:
            return "break"
        idx = self._current_index()
        nxt = 0 if idx is None else max(
            0, min(len(self.project.cues) - 1, idx + delta)
        )
        self.cue_list.select(nxt)
        self.cue_list.focus_set()
        self._play_selected()
        return "break"

    # ------------------------------------------------------------- tail fix

    def _fix_tail_clipping(self) -> str:
        if self.project is None:
            messagebox.showinfo("프로젝트 없음", "먼저 MP3로 새 프로젝트를 만드세요.")
            return "break"
        if self._tail_busy:
            return "break"
        if not self.project.cues:
            messagebox.showinfo("문장 없음", "보정할 문장이 없습니다.")
            return "break"

        self._stop_playback(quiet=True)
        self._tail_busy = True
        self._tail_project_dir = self.project.directory
        if hasattr(self, "tail_btn"):
            self.tail_btn.configure(state="disabled")
        self.status_var.set("끝 잘림 분석 중…")

        result_queue: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._tail_queue = result_queue
        spans = [
            CueSpan(index=cue.index, start=cue.start, end=cue.end)
            for cue in self.project.cues
        ]
        worker = threading.Thread(
            target=self._tail_worker,
            args=(self.project.audio, spans, result_queue),
            daemon=True,
        )
        worker.start()
        self.after(100, lambda: self._poll_tail(result_queue))
        return "break"

    @staticmethod
    def _tail_worker(
        audio_path: Path,
        spans: list[CueSpan],
        result_queue: "queue.Queue[tuple[str, object]]",
    ) -> None:
        try:
            from .audio import decode_audio

            audio = decode_audio(audio_path)
            fixes = analyze_tail_clipping(audio, spans)
            result_queue.put(("ok", fixes))
        except Exception as exc:  # noqa: BLE001 - GUI에 오류를 표시
            result_queue.put(("error", (str(exc), traceback.format_exc())))

    def _poll_tail(self, result_queue: "queue.Queue[tuple[str, object]]") -> None:
        if self._tail_queue is not result_queue:
            return
        try:
            kind, payload = result_queue.get_nowait()
        except queue.Empty:
            self.after(120, lambda: self._poll_tail(result_queue))
            return

        self._tail_queue = None
        self._tail_busy = False
        if hasattr(self, "tail_btn"):
            self.tail_btn.configure(state="normal")

        if kind == "error":
            self.status_var.set("끝 잘림 분석 실패")
            message, detail = payload  # type: ignore[misc]
            messagebox.showerror(
                "끝 잘림 분석 실패",
                str(message) + ("\n\n" + str(detail)[-1200:] if detail else ""),
            )
            return

        if self.project is None or self.project.directory != self._tail_project_dir:
            self.status_var.set("끝 잘림 분석 취소됨")
            return

        fixes: list[TailFix] = list(payload)  # type: ignore[arg-type]
        if not fixes:
            self.status_var.set("끝 잘림 없음")
            messagebox.showinfo(
                "끝 잘림 보정",
                "끝이 미세하게 잘린 문장을 찾지 못했습니다.",
            )
            return

        by_index = {fix.index: fix for fix in fixes}
        start_shifts = {
            fix.next_index: fix.next_new_start
            for fix in fixes
            if fix.next_index is not None and fix.next_new_start is not None
        }
        for cue in self.project.cues:
            fix = by_index.get(cue.index)
            if fix is not None:
                cue.end = fix.new_end
            new_start = start_shifts.get(cue.index)
            if new_start is not None:
                cue.start = new_start

        try:
            saved_paths = save_project_cues(self.project)
        except Exception as exc:  # noqa: BLE001 - 저장 실패를 사용자에게 알림
            messagebox.showerror(
                "저장 실패",
                f"종료 시간을 저장하지 못했습니다.\n{exc}",
            )
            self._reload()
            return

        selected = self._current_index()
        self._populate(selected)
        self.status_var.set(f"끝 잘림 {len(fixes)}개 보정 완료")
        TailFixReport(self, fixes=fixes, saved_paths=saved_paths)

    # -------------------------------------------------------------- project

    def _choose_project(self) -> None:
        initial = (
            self.project.directory.parent
            if self.project is not None
            else default_project_root()
        )
        path = filedialog.askdirectory(
            title="프로젝트 폴더 선택",
            initialdir=str(initial.resolve()),
        )
        if not path:
            return
        try:
            project = load_project(path)
        except Exception as exc:
            messagebox.showerror("프로젝트 열기 실패", str(exc))
            return
        self._set_project(project)

    def _reload(self) -> None:
        if self.project is None:
            self._choose_project()
            return
        try:
            project = load_project(self.project.directory)
        except Exception as exc:
            messagebox.showerror("새로고침 실패", str(exc))
            return
        self._set_project(project)

    def _new_project(self) -> None:
        path = filedialog.askopenfilename(
            parent=self,
            title="MP3 파일 선택",
            filetypes=[
                ("오디오 파일", "*.mp3 *.wav *.m4a *.flac *.ogg"),
                ("모든 파일", "*.*"),
            ],
        )
        if not path:
            return
        audio_path = Path(path)
        dialog = NewProjectDialog(self, audio_path)
        if dialog.result is None:
            return
        name = dialog.result["name"]
        model = dialog.result["model"]

        project_dir = self._unique_project_dir(audio_path)
        try:
            project_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("폴더 생성 실패", str(exc))
            return

        self.new_btn.configure(state="disabled")
        progress = ProgressDialog(
            self,
            note=(
                f"{audio_path.name}\n"
                "NLLB 번역 · 최초 1회 모델 다운로드 후 로컬 캐시 재사용"
            ),
        )
        worker = threading.Thread(
            target=self._project_worker,
            args=(audio_path, project_dir, name, model, progress),
            daemon=True,
        )
        worker.start()
        self.after(120, lambda: self._poll_project(progress))

    def _unique_project_dir(self, audio_path: Path) -> Path:
        base = default_project_root()
        base.mkdir(parents=True, exist_ok=True)
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", audio_path.stem).strip(" .")
        cleaned = cleaned or "project"
        candidate = base / cleaned
        if (
            candidate.exists()
            and not (candidate / "project.json").exists()
            and (candidate / audio_path.name).exists()
            and (candidate / audio_path.name).stat().st_size == audio_path.stat().st_size
        ):
            # 이전에 중단/실패한 작업이면 같은 폴더를 써서 ASR 캐시를 재사용합니다.
            return candidate
        n = 2
        while candidate.exists():
            candidate = base / f"{cleaned}_{n}"
            n += 1
        return candidate

    def _project_worker(
        self,
        audio_path: Path,
        project_dir: Path,
        name: str,
        model: str,
        dialog: ProgressDialog,
    ) -> None:
        from .pipeline import CancelledError, RunConfig, run

        try:
            project_dir.mkdir(parents=True, exist_ok=True)
            target = project_dir / audio_path.name
            shutil.copy2(audio_path, target)

            try:
                from .asr import device_note

                dialog.push("note", (device_note(),))
            except Exception:
                pass

            config = RunConfig(
                input_path=target,
                out_dir=project_dir,
                model=model,
                translator="nllb",
                formats=("srt", "vtt", "txt", "md", "csv", "json"),
                title=name,
            )
            run(
                config,
                progress=lambda message, fraction: dialog.push(
                    "progress", (message, fraction)
                ),
                cancel=dialog.cancel_event.is_set,
            )
            create_project(
                project_dir,
                name=name,
                audio=target,
                out_dir=project_dir,
                move=False,
            )
            dialog.push("done", (str(project_dir),))
        except CancelledError:
            dialog.push("cancelled", (str(project_dir),))
        except Exception as exc:  # noqa: BLE001 - GUI에 오류를 표시
            dialog.push("error", (str(exc), traceback.format_exc()))

    def _poll_project(self, dialog: ProgressDialog) -> None:
        finished: str | None = None
        payload: tuple = ()
        while True:
            try:
                kind, data = dialog.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                message, fraction = data
                dialog.set_progress(message, fraction)
            elif kind == "note":
                dialog.set_note(data[0])
            else:
                finished, payload = kind, data
                break

        if finished is None:
            if dialog.winfo_exists():
                self.after(120, lambda: self._poll_project(dialog))
            return

        dialog.close()
        self.new_btn.configure(state="normal")
        if finished == "done":
            try:
                project = load_project(payload[0])
                self.projects = discover_projects()
                self._set_project(project)
                self.status_var.set("새 프로젝트 생성 완료")
            except Exception as exc:
                messagebox.showerror("프로젝트 열기 실패", str(exc))
        elif finished == "cancelled":
            self.status_var.set("프로젝트 생성 취소됨")
            messagebox.showinfo(
                "취소됨",
                "프로젝트 생성이 취소되었습니다.\n"
                "인식 캐시는 저장되어 있어 다시 만들 때 재사용됩니다.",
            )
        else:
            message = payload[0] if payload else "알 수 없는 오류"
            detail = payload[1] if len(payload) > 1 else ""
            messagebox.showerror(
                "프로젝트 생성 실패", message + ("\n\n" + detail[-1500:] if detail else "")
            )

    def _set_project(self, project: Project) -> None:
        # 프로젝트를 바꾸기 전에 현재 목록 위치를 확정해서 저장합니다.
        self._flush_position()
        self._stop_playback(quiet=True)
        self.project = project
        self.show_korean = True
        self.mode_var.set("표시: 한영")
        self.header_name.set(project.name)
        self.header_sub.set(self._project_subtitle(project))
        self._populate(self._saved_index(project))
        self.cue_list.focus_set()

    def _on_close(self) -> None:
        self._flush_position()
        self._stop_playback(quiet=True)
        if self.player is not None:
            self.player.close()
        self.destroy()


def main(argv: list[str] | None = None) -> int:
    _enable_dpi_awareness()
    parser = argparse.ArgumentParser(description="영어 듣기 학습 GUI")
    parser.add_argument(
        "--project",
        help="프로젝트 폴더 (project.json이 있는 폴더). 생략하면 data/project에서 찾습니다.",
    )
    parser.add_argument(
        "--theme",
        choices=("light", "dark"),
        default="light",
        help="GUI 테마 (기본: light). 실행 중 Ctrl+D로 전환할 수 있습니다.",
    )
    args = parser.parse_args(argv)

    try:
        if args.project:
            project = load_project(args.project)
            projects = [project]
        else:
            projects = discover_projects()
            project = projects[0] if projects else None
    except Exception as exc:
        root = ctk.CTk()
        root.withdraw()
        messagebox.showerror("프로젝트를 열 수 없습니다", str(exc))
        root.destroy()
        return 2

    app = StudyApp(project, projects, theme=args.theme)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
