"""영어 듣기 학습 GUI.

왼쪽 문장 목록에서 항목을 고르고 Space로 해당 구간을 재생하며,
Enter로 오른쪽 화면의 한국어 번역을 보이거나 숨깁니다.
"""

from __future__ import annotations

import argparse
import io
import os
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
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Callable

from .audio import find_ffmpeg
from .project import (
    Project,
    create_project,
    default_project_root,
    discover_projects,
    load_project,
)
from .subtitles import format_mmss


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


class CueList(tk.Frame):
    """문장이 줄바꿈되어 표시되는 스크롤 목록.

    tk.Listbox는 긴 문장을 한 줄로 잘라 버리므로, 읽기 전용 Text 위젯에
    문장 카드를 쌓아 전체 내용이 보이도록 했습니다.
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        font_head: tuple,
        font_body: tuple,
        on_select: "Callable[[int], None]",
        on_play: "Callable[[], None]",
    ) -> None:
        super().__init__(master, bg="#ffffff")
        self._on_select = on_select
        self._on_play = on_play
        self._cues: list = []
        self._ranges: list[tuple[str, str]] = []
        self._current: int | None = None

        scroll = ttk.Scrollbar(self, orient="vertical")
        scroll.pack(side="right", fill="y")
        self.text = tk.Text(
            self,
            wrap="word",
            bd=0,
            relief="flat",
            highlightthickness=0,
            padx=10,
            pady=4,
            cursor="arrow",
            takefocus=1,
            font=font_body,
            yscrollcommand=scroll.set,
        )
        self.text.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.text.yview)

        self.text.tag_configure("head", font=font_head, foreground="#64748b", spacing1=9)
        self.text.tag_configure(
            "body", font=font_body, foreground="#1e293b", lmargin1=10, lmargin2=10, spacing3=4
        )
        self.text.tag_configure("q", foreground="#1d4ed8")
        self.text.tag_configure("ko", foreground="#b45309")
        self.text.tag_configure("sel", background="#dbeafe")
        self.text.tag_raise("sel")
        self.text.configure(state="disabled")

        self.text.bind("<Button-1>", self._on_click)
        self.text.bind("<Double-Button-1>", self._on_double)
        self.text.bind("<Up>", lambda _e: self._move(-1))
        self.text.bind("<Down>", lambda _e: self._move(1))
        self.text.bind("<Prior>", lambda _e: self._move(-8))
        self.text.bind("<Next>", lambda _e: self._move(8))
        self.text.bind("<Home>", lambda _e: self._jump(0))
        self.text.bind("<End>", lambda _e: self._jump(len(self._cues) - 1))

    # -- list-like API -----------------------------------------------------

    def set_cues(self, cues: list) -> None:
        self._cues = list(cues)
        self._ranges = []
        self._current = None
        widget = self.text
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        for i, cue in enumerate(self._cues):
            start = widget.index("end-1c")
            head = f"{cue.index:03d}   [{format_mmss(cue.start)}]"
            if cue.question:
                head += f"   Q{cue.question}"
            widget.insert("end", head + "\n", ("head", f"cue{i}", "q" if cue.question else ""))
            body = (cue.text or "").strip()
            if cue.lang == "ko":
                body = "[한국어] " + body
            widget.insert("end", body + "\n\n", ("body", f"cue{i}", "ko" if cue.lang == "ko" else ""))
            self._ranges.append((start, widget.index("end-1c")))
        widget.configure(state="disabled")
        widget.yview_moveto(0)

    def size(self) -> int:
        return len(self._cues)

    def current_index(self) -> int | None:
        return self._current

    def select(self, index: int, *, see: bool = True) -> None:
        if not 0 <= index < len(self._cues):
            return
        self._current = index
        widget = self.text
        widget.tag_remove("sel", "1.0", "end")
        start, end = self._ranges[index]
        widget.tag_add("sel", start, end)
        widget.tag_raise("sel")
        if see:
            widget.see(start)
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


class NewProjectDialog(tk.Toplevel):
    """MP3를 선택한 뒤 프로젝트 이름/모델을 정하는 작은 창."""

    def __init__(self, master: tk.Misc, audio_path: Path) -> None:
        super().__init__(master)
        self.result: dict[str, str] | None = None
        self.title("새 프로젝트 만들기")
        self.resizable(False, False)
        self.transient(master)
        self.configure(bg="#ffffff")

        body = tk.Frame(self, bg="#ffffff", padx=24, pady=20)
        body.pack(fill="both", expand=True)

        tk.Label(
            body,
            text="MP3로 새 프로젝트",
            font=(tkfont.nametofont("TkDefaultFont").actual("family"), 15, "bold"),
            bg="#ffffff",
            fg="#0f172a",
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            body,
            text=audio_path.name,
            font=(tkfont.nametofont("TkDefaultFont").actual("family"), 10),
            bg="#ffffff",
            fg="#64748b",
            anchor="w",
        ).pack(fill="x", pady=(2, 16))

        form = tk.Frame(body, bg="#ffffff")
        form.pack(fill="x")
        tk.Label(
            form, text="프로젝트 이름", bg="#ffffff", fg="#334155", anchor="w"
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.name_var = tk.StringVar(value=audio_path.stem)
        entry = ttk.Entry(form, textvariable=self.name_var, width=36)
        entry.grid(row=0, column=1, sticky="ew", padx=(14, 0), pady=(0, 6))
        form.columnconfigure(1, weight=1)

        tk.Label(
            form, text="Whisper 모델", bg="#ffffff", fg="#334155", anchor="w"
        ).grid(row=1, column=0, sticky="w", pady=6)
        self.model_var = tk.StringVar(value="small")
        model_box = ttk.Combobox(
            form,
            textvariable=self.model_var,
            values=("base", "small", "medium"),
            state="readonly",
            width=12,
        )
        model_box.grid(row=1, column=1, sticky="w", padx=(14, 0), pady=6)

        tk.Label(
            form, text="번역", bg="#ffffff", fg="#334155", anchor="w"
        ).grid(row=2, column=0, sticky="w", pady=6)
        tk.Label(
            form,
            text="NLLB (오프라인)  ·  Google 번역 사용 안 함",
            bg="#ffffff",
            fg="#1d4ed8",
            anchor="w",
        ).grid(row=2, column=1, sticky="w", padx=(14, 0), pady=6)

        tk.Label(
            body,
            text=(
                "NLLB 모델은 최초 1회만 내려받고(약 2.4GB) 이후 로컬 캐시를 재사용합니다.\n"
                "CPU에서 small 모델은 22분 음원 기준 대략 15~20분, 번역에 몇 분 더 걸릴 수 있습니다."
            ),
            font=(tkfont.nametofont("TkDefaultFont").actual("family"), 9),
            bg="#f8fafc",
            fg="#64748b",
            justify="left",
            anchor="w",
            padx=10,
            pady=8,
        ).pack(fill="x", pady=(18, 14))

        buttons = tk.Frame(body, bg="#ffffff")
        buttons.pack(fill="x")
        ttk.Button(buttons, text="취소", command=self.destroy).pack(side="right")
        ttk.Button(buttons, text="만들기", command=self._ok).pack(
            side="right", padx=(0, 8)
        )

        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.grab_set()
        self.update_idletasks()
        width = max(460, self.winfo_reqwidth())
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


class ProgressDialog(tk.Toplevel):
    """프로젝트 생성 진행률을 보여주는 창."""

    def __init__(self, master: tk.Misc, *, note: str) -> None:
        super().__init__(master)
        self.title("프로젝트 생성 중")
        self.resizable(False, False)
        self.transient(master)
        self.configure(bg="#ffffff")
        self.queue: "queue.Queue[tuple[str, tuple]]" = queue.Queue()
        self.cancel_event = threading.Event()
        self.started = time.time()
        self._closed = False

        body = tk.Frame(self, bg="#ffffff", padx=26, pady=22)
        body.pack(fill="both", expand=True)
        tk.Label(
            body,
            text="프로젝트를 만들고 있습니다",
            font=(tkfont.nametofont("TkDefaultFont").actual("family"), 15, "bold"),
            bg="#ffffff",
            fg="#0f172a",
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            body,
            text=note,
            font=(tkfont.nametofont("TkDefaultFont").actual("family"), 9),
            bg="#ffffff",
            fg="#64748b",
            anchor="w",
            justify="left",
        ).pack(fill="x", pady=(4, 16))

        self.status_var = tk.StringVar(value="준비 중…")
        tk.Label(
            body,
            textvariable=self.status_var,
            font=(tkfont.nametofont("TkDefaultFont").actual("family"), 10),
            bg="#ffffff",
            fg="#334155",
            anchor="w",
        ).pack(fill="x")
        self.bar = ttk.Progressbar(body, mode="determinate", maximum=100, length=460)
        self.bar.pack(fill="x", pady=(8, 6))
        self.elapsed_var = tk.StringVar(value="경과 00:00")
        row = tk.Frame(body, bg="#ffffff")
        row.pack(fill="x")
        tk.Label(
            row,
            textvariable=self.elapsed_var,
            font=(tkfont.nametofont("TkDefaultFont").actual("family"), 9),
            bg="#ffffff",
            fg="#94a3b8",
        ).pack(side="left")
        self.cancel_btn = ttk.Button(row, text="취소", command=self.request_cancel)
        self.cancel_btn.pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self.request_cancel)
        self.grab_set()
        self.update_idletasks()
        width = max(520, self.winfo_reqwidth())
        height = self.winfo_reqheight()
        x = master.winfo_rootx() + max(0, (master.winfo_width() - width) // 2)
        y = master.winfo_rooty() + max(0, (master.winfo_height() - height) // 3)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.after(300, self._tick)

    def push(self, kind: str, payload: tuple) -> None:
        self.queue.put((kind, payload))

    def set_progress(self, message: str, fraction: float | None) -> None:
        self.status_var.set(message)
        if fraction is None:
            if str(self.bar.cget("mode")) != "indeterminate":
                self.bar.configure(mode="indeterminate")
                self.bar.start(14)
            return
        if str(self.bar.cget("mode")) != "determinate":
            self.bar.stop()
            self.bar.configure(mode="determinate")
        self.bar["value"] = max(0.0, min(100.0, fraction * 100.0))

    def request_cancel(self) -> None:
        if self.cancel_event.is_set():
            return
        self.cancel_event.set()
        self.cancel_btn.state(["disabled"])
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


class StudyApp(tk.Tk):
    LEFT_WIDTH = 500

    def __init__(
        self, project: Project | None, projects: list[Project] | None = None
    ) -> None:
        _enable_dpi_awareness()
        super().__init__()
        self.project = project
        if projects is not None:
            self.projects = projects
        else:
            self.projects = [project] if project is not None else []
        self.show_korean = True
        self.playing_index: int | None = None
        self._stop_after: str | None = None
        self.player: SegmentPlayer | None = None
        self._player_error: str | None = None

        self.title(f"{project.name} — 영어 듣기 학습" if project else "영어 듣기 학습")
        self.geometry(self._center_geometry(1180, 760))
        self.minsize(900, 560)
        self.configure(bg="#f1f5f9")

        self._init_fonts()
        self._init_player()
        self._build_ui()
        self._bind_keys()
        self._populate()
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
        self.font_title = (korean, 19, "bold")
        self.font_sub = (korean, 10)
        self.font_list = (mono, 10)
        self.font_card = (korean, 11)
        self.font_english = (body, 20)
        self.font_korean = (korean, 14)
        self.font_small = (korean, 10)
        self.font_info = (korean, 11, "bold")

    def _init_player(self) -> None:
        try:
            self.player = SegmentPlayer()
        except PlaybackError as exc:
            self._player_error = str(exc)
            self.player = None

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", font=(self.font_small[0], 10), padding=(12, 6))
        style.configure("TCheckbutton", background="#ffffff")

        header = tk.Frame(self, bg="#0f172a")
        header.pack(side="top", fill="x")
        self.header_name = tk.StringVar(
            value=self.project.name if self.project else "새 프로젝트"
        )
        tk.Label(
            header,
            textvariable=self.header_name,
            font=self.font_title,
            bg="#0f172a",
            fg="#f8fafc",
            anchor="w",
        ).pack(side="left", padx=(22, 12), pady=(14, 2))
        self.header_sub = tk.StringVar(
            value=(
                self._project_subtitle(self.project)
                if self.project
                else "오른쪽 위 [+ MP3로 새 프로젝트]를 눌러 시작하세요"
            )
        )
        tk.Label(
            header,
            textvariable=self.header_sub,
            font=self.font_sub,
            bg="#0f172a",
            fg="#94a3b8",
            anchor="w",
        ).pack(side="left", pady=(18, 2))
        self.new_btn = ttk.Button(
            header, text="+ MP3로 새 프로젝트", command=self._new_project
        )
        self.new_btn.pack(side="right", padx=(6, 16), pady=12)
        ttk.Button(
            header, text="프로젝트 폴더 열기", command=self._open_folder
        ).pack(side="right", padx=(6, 6), pady=12)

        # 하단 바를 body보다 먼저 pack해야 창 높이가 줄어들 때도 보입니다.
        bar = tk.Frame(self, bg="#e2e8f0")
        bar.pack(side="bottom", fill="x")
        self.play_btn = ttk.Button(bar, text="▶ 재생 (Space)", command=self._play_selected)
        self.play_btn.pack(side="left", padx=(16, 6), pady=8)
        ttk.Button(bar, text="■ 정지 (Esc)", command=self._stop_playback).pack(
            side="left", padx=6, pady=8
        )
        ttk.Button(bar, text="이전 문장", command=lambda: self._step(-1)).pack(
            side="left", padx=(18, 6), pady=8
        )
        ttk.Button(bar, text="다음 문장", command=lambda: self._step(1)).pack(
            side="left", padx=6, pady=8
        )
        self.mode_var = tk.StringVar(value="표시: 한영")
        tk.Label(
            bar,
            textvariable=self.mode_var,
            font=self.font_small,
            bg="#e2e8f0",
            fg="#475569",
        ).pack(side="left", padx=18)

        self.status_var = tk.StringVar(value="준비됨")
        tk.Label(
            bar,
            textvariable=self.status_var,
            font=self.font_small,
            bg="#e2e8f0",
            fg="#334155",
            anchor="e",
        ).pack(side="right", padx=16)

        if self._player_error:
            self.status_var.set("오디오 재생 불가")
            self.play_btn.state(["disabled"])

        body = tk.Frame(self, bg="#f1f5f9")
        body.pack(side="top", fill="both", expand=True, padx=14, pady=12)

        paned = tk.PanedWindow(
            body, orient="horizontal", sashwidth=8, bg="#f1f5f9", bd=0
        )
        paned.pack(fill="both", expand=True)

        left = tk.Frame(paned, bg="#ffffff", highlightbackground="#e2e8f0", highlightthickness=1)
        right = tk.Frame(paned, bg="#ffffff", highlightbackground="#e2e8f0", highlightthickness=1)
        paned.add(left, minsize=340, width=self.LEFT_WIDTH)
        paned.add(right, minsize=420, width=680)

        # left: sentence list
        list_head = tk.Frame(left, bg="#ffffff")
        list_head.pack(fill="x", padx=16, pady=(14, 6))
        tk.Label(
            list_head,
            text="문장 목록",
            font=(self.font_info[0], 12, "bold"),
            bg="#ffffff",
            fg="#0f172a",
        ).pack(side="left")
        tk.Label(
            list_head,
            text="Space 재생",
            font=self.font_small,
            bg="#ffffff",
            fg="#94a3b8",
        ).pack(side="right")

        self.cue_list = CueList(
            left,
            font_head=(self.font_list[0], 9),
            font_body=self.font_card,
            on_select=self._on_select,
            on_play=self._play_selected,
        )
        self.cue_list.pack(fill="both", expand=True, padx=(12, 8), pady=(0, 12))

        # right: study view
        view = tk.Frame(right, bg="#ffffff")
        view.pack(fill="both", expand=True, padx=34, pady=26)

        self.info_var = tk.StringVar(value="문장을 선택하세요")
        tk.Label(
            view,
            textvariable=self.info_var,
            font=self.font_info,
            bg="#ffffff",
            fg="#64748b",
            anchor="w",
        ).pack(fill="x")

        tk.Frame(view, bg="#e2e8f0", height=1).pack(fill="x", pady=(10, 22))

        self.english_label = tk.Label(
            view,
            text="",
            font=self.font_english,
            bg="#ffffff",
            fg="#0f172a",
            justify="left",
            anchor="nw",
            wraplength=620,
        )
        self.english_label.pack(fill="x", anchor="w")

        self.korean_label = tk.Label(
            view,
            text="",
            font=self.font_korean,
            bg="#ffffff",
            fg="#475569",
            justify="left",
            anchor="nw",
            wraplength=620,
        )
        self.korean_label.pack(fill="x", anchor="w", pady=(18, 0))

        tk.Frame(view, bg="#ffffff").pack(fill="both", expand=True)
        self.hint_label = tk.Label(
            view,
            text="Space: 재생/정지   ·   Enter: 한영 전환   ·   j/k: 위/아래 이동 + 재생   ·   Esc: 정지",
            font=self.font_small,
            bg="#ffffff",
            fg="#94a3b8",
            anchor="w",
        )
        self.hint_label.pack(fill="x", side="bottom")

        view.bind("<Configure>", self._on_view_resize)

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
        self.bind("<F5>", lambda _e: self._reload())

    # ------------------------------------------------------------- contents

    def _populate(self) -> None:
        cues = self.project.cues if self.project else []
        self.cue_list.set_cues(cues)
        self.title(f"{self.project.name} — 영어 듣기 학습" if self.project else "영어 듣기 학습")
        if cues:
            self.cue_list.select(0)
        self._update_view()

    def _current_index(self) -> int | None:
        return self.cue_list.current_index()

    def _update_view(self) -> None:
        idx = self._current_index()
        if idx is None or self.project is None:
            self.info_var.set(
                "문장을 선택하세요" if self.project else "먼저 프로젝트를 만드세요"
            )
            self.english_label.configure(text="", fg="#0f172a")
            self.korean_label.configure(
                text=(
                    "오른쪽 위 [+ MP3로 새 프로젝트]를 눌러 MP3를 고르면\n"
                    "자막을 만들고 바로 공부할 수 있습니다."
                    if self.project is None
                    else ""
                ),
                fg="#94a3b8",
            )
            return

        cue = self.project.cues[idx]
        question = f"  ·  {cue.question}번 문제" if cue.question else ""
        self.info_var.set(
            f"문장 {cue.index:03d} / {len(self.project.cues):03d}"
            f"   ·   {format_mmss(cue.start)}–{format_mmss(cue.end)}{question}"
        )

        if cue.lang == "ko":
            self.english_label.configure(text=cue.text, fg="#b45309")
            self.korean_label.configure(
                text="한국어 안내 문장입니다.", fg="#94a3b8"
            )
            return

        self.english_label.configure(text=cue.text, fg="#0f172a")
        if self.show_korean:
            self.korean_label.configure(
                text=cue.translation or "(번역 없음)", fg="#475569"
            )
        else:
            self.korean_label.configure(
                text="한국어 번역 숨김  ·  Enter로 다시 표시", fg="#cbd5e1"
            )

    def _on_view_resize(self, event: tk.Event) -> None:
        width = max(280, int(event.width) - 20)
        self.english_label.configure(wraplength=width)
        self.korean_label.configure(wraplength=width)

    def _on_select(self, _index: int) -> None:
        self._update_view()

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
            end = min(end, max(cue.start + 0.25, next_start - 0.03))
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

    # -------------------------------------------------------------- project

    def _open_folder(self) -> None:
        if self.project is None:
            messagebox.showinfo("프로젝트 없음", "먼저 MP3로 새 프로젝트를 만드세요.")
            return
        path = str(self.project.directory.resolve())
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            messagebox.showerror("폴더 열기 실패", str(exc))

    def _choose_project(self) -> None:
        path = filedialog.askdirectory(
            title="프로젝트 폴더 선택",
            initialdir=str(default_project_root().resolve()),
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

        self.new_btn.state(["disabled"])
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
            else:
                finished, payload = kind, data
                break

        if finished is None:
            if dialog.winfo_exists():
                self.after(120, lambda: self._poll_project(dialog))
            return

        dialog.close()
        self.new_btn.state(["!disabled"])
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
        self._stop_playback(quiet=True)
        self.project = project
        self.show_korean = True
        self.mode_var.set("표시: 한영")
        self.header_name.set(project.name)
        self.header_sub.set(self._project_subtitle(project))
        self._populate()
        self.cue_list.focus_set()

    def _on_close(self) -> None:
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
    args = parser.parse_args(argv)

    try:
        if args.project:
            project = load_project(args.project)
            projects = [project]
        else:
            projects = discover_projects()
            project = projects[0] if projects else None
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("프로젝트를 열 수 없습니다", str(exc))
        root.destroy()
        return 2

    app = StudyApp(project, projects)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
