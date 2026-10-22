"""
app.py — brutalist desktop GUI for the document-search RAG app.

A deliberately raw, high-contrast interface: thick black borders, monospace
type, uppercase labels, flat blocks, one harsh accent colour. Built with raw
Tk widgets (not ttk) so every border, colour and relief is under our control.

Functionally identical to before:
  1. Point at a folder or files of your own documents and index them.
  2. Ask questions in natural language.
  3. See a synthesised answer plus the exact source passages it came from.

Everything runs locally. An LLM is used only if you enter an API key in
Settings; otherwise answers are extracted directly from your documents.
"""

from __future__ import annotations

import os
import sys
import json
import ctypes
import threading
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox


def enable_dpi_awareness() -> None:
    """Tell Windows this process is DPI-aware so the UI renders crisp (not a
    blurry stretched bitmap) on scaled displays (125% / 150% / 200%)."""
    if sys.platform != "win32":
        return
    try:
        # per-monitor DPI aware (best); falls back for older Windows
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

from rag_core import RagIndex, answer_query, SUPPORTED_EXTS

APP_NAME = "DOCFINDER // RAG"
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".docfinder_rag")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
INDEX_PATH = os.path.join(CONFIG_DIR, "index.pkl")

# --- brutalist palette --------------------------------------------------- #
BG = "#E8E6DE"        # raw paper / concrete
INK = "#111111"       # near-black
PANEL = "#FFFFFF"     # stark white panels
ACCENT = "#FFDD00"    # harsh yellow
ACCENT2 = "#FF4B2B"   # alarm red
MUTE = "#6B6B6B"

MONO = ("Consolas", 10)
MONO_B = ("Consolas", 10, "bold")
MONO_BIG = ("Consolas", 13, "bold")
MONO_SM = ("Consolas", 9)


def load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)


# --- brutalist widget factory -------------------------------------------- #
def block_label(parent, text, bg=INK, fg=PANEL, font=MONO_B, **kw):
    return tk.Label(
        parent, text=text, bg=bg, fg=fg, font=font,
        bd=0, padx=10, pady=4, anchor="w", **kw,
    )


def hard_button(parent, text, command, bg=PANEL, fg=INK):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=fg, font=MONO_B,
        activebackground=ACCENT, activeforeground=INK,
        bd=0, highlightthickness=3, highlightbackground=INK,
        highlightcolor=INK, relief="flat", padx=12, pady=5,
        cursor="hand2",
    )

    def on_enter(_e):
        btn.configure(bg=ACCENT)

    def on_leave(_e):
        btn.configure(bg=bg)

    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)
    return btn


def hard_entry(parent, textvariable):
    return tk.Entry(
        parent, textvariable=textvariable, font=MONO,
        bg=PANEL, fg=INK, insertbackground=INK,
        bd=0, highlightthickness=3, highlightbackground=INK,
        highlightcolor=ACCENT2, relief="flat",
    )


def framed(parent, **kw):
    """A block with a thick black border."""
    return tk.Frame(
        parent, bg=PANEL, bd=0,
        highlightthickness=3, highlightbackground=INK, highlightcolor=INK, **kw,
    )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1040x720")
        self.minsize(820, 560)
        self.configure(bg=BG)

        self.config_data = load_config()
        self.index: RagIndex | None = None
        self._busy = False

        # scale Tk to the real screen DPI so fonts stay sharp and correctly sized
        try:
            dpi = self.winfo_fpixels("1i")  # pixels per inch (real, once DPI-aware)
            if dpi > 0:
                self.tk.call("tk", "scaling", dpi / 72.0)
        except Exception:
            pass

        self._build_ui()
        self._try_load_saved_index()

    # ------------------------------------------------------------------ UI #
    def _build_ui(self):
        pad = {"padx": 14, "pady": (14, 0)}

        # masthead
        head = tk.Frame(self, bg=INK)
        head.pack(fill="x", padx=14, pady=(14, 0))
        tk.Label(
            head, text=" DOCFINDER ", bg=ACCENT, fg=INK, font=("Consolas", 18, "bold"),
            padx=10, pady=6,
        ).pack(side="left", padx=(6, 0), pady=6)
        tk.Label(
            head, text="RETRIEVAL-AUGMENTED DOCUMENT SEARCH // OFFLINE",
            bg=INK, fg=PANEL, font=MONO_B, padx=12,
        ).pack(side="left", pady=6)

        # source row
        srow = tk.Frame(self, bg=BG)
        srow.pack(fill="x", **pad)
        block_label(srow, "DOCS").pack(side="left")
        self.source_var = tk.StringVar(value=self.config_data.get("last_source", ""))
        e = hard_entry(srow, self.source_var)
        e.pack(side="left", fill="x", expand=True, padx=8)
        hard_button(srow, "FOLDER", self.pick_folder).pack(side="left", padx=(0, 6))
        hard_button(srow, "FILES", self.pick_files).pack(side="left", padx=(0, 6))
        hard_button(srow, "BUILD INDEX", self.build_index, bg=ACCENT).pack(side="left", padx=(0, 6))
        hard_button(srow, "SETTINGS", self.open_settings).pack(side="left")

        # status strip
        self.status_var = tk.StringVar(value="NO INDEX. CHOOSE DOCS > BUILD INDEX.")
        self.status_lbl = tk.Label(
            self, textvariable=self.status_var, bg=INK, fg=ACCENT,
            font=MONO_SM, anchor="w", padx=10, pady=4,
        )
        self.status_lbl.pack(fill="x", padx=14, pady=(10, 0))

        # query row
        qrow = tk.Frame(self, bg=BG)
        qrow.pack(fill="x", **pad)
        block_label(qrow, "ASK", bg=ACCENT2, fg=PANEL).pack(side="left")
        self.query_var = tk.StringVar()
        qe = hard_entry(qrow, self.query_var)
        qe.pack(side="left", fill="x", expand=True, padx=8)
        qe.bind("<Return>", lambda _e: self.ask())
        block_label(qrow, "TOP").pack(side="left")
        self.topk_var = tk.IntVar(value=5)
        tk.Spinbox(
            qrow, from_=1, to=20, width=3, textvariable=self.topk_var,
            font=MONO_B, bg=PANEL, fg=INK, bd=0,
            highlightthickness=3, highlightbackground=INK, relief="flat",
            buttonbackground=INK,
        ).pack(side="left", padx=8)
        self.ask_btn = hard_button(qrow, "SEARCH >>", self.ask, bg=ACCENT)
        self.ask_btn.pack(side="left")

        # answer block
        ans_wrap = tk.Frame(self, bg=BG)
        ans_wrap.pack(fill="both", expand=False, padx=14, pady=(14, 0))
        block_label(ans_wrap, "// ANSWER", bg=INK, fg=ACCENT, font=MONO_B).pack(fill="x")
        abox = framed(ans_wrap)
        abox.pack(fill="both", expand=True)
        self.answer_box = tk.Text(
            abox, height=6, wrap="word", font=("Consolas", 11),
            bg=PANEL, fg=INK, bd=0, padx=10, pady=8, relief="flat",
            insertbackground=INK,
        )
        self.answer_box.pack(fill="both", expand=True)
        self.answer_box.configure(state="disabled")

        # sources block
        src_wrap = tk.Frame(self, bg=BG)
        src_wrap.pack(fill="both", expand=True, padx=14, pady=(14, 14))
        block_label(src_wrap, "// SOURCE PASSAGES", bg=INK, fg=ACCENT, font=MONO_B).pack(fill="x")
        sbox = framed(src_wrap)
        sbox.pack(fill="both", expand=True)
        self.sources_box = tk.Text(
            sbox, wrap="word", font=MONO_SM, bg=PANEL, fg=INK,
            bd=0, padx=10, pady=8, relief="flat", insertbackground=INK,
        )
        scroll = tk.Scrollbar(
            sbox, command=self.sources_box.yview, bg=INK, troughcolor=BG,
            bd=0, relief="flat", activebackground=ACCENT, width=14,
        )
        self.sources_box.configure(yscrollcommand=scroll.set, state="disabled")
        scroll.pack(side="right", fill="y")
        self.sources_box.pack(fill="both", expand=True)
        self.sources_box.tag_configure("src", foreground=INK, background=ACCENT, font=MONO_B)
        self.sources_box.tag_configure("score", foreground=ACCENT2, font=MONO_B)
        self.sources_box.tag_configure("path", foreground=MUTE, font=MONO_SM)

    # -------------------------------------------------------------- helpers #
    def _set_text(self, widget: tk.Text, content: str):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _set_busy(self, busy: bool, msg: str | None = None):
        self._busy = busy
        if msg:
            self.status_var.set(msg.upper())
        self.ask_btn.configure(state="disabled" if busy else "normal")
        self.status_lbl.configure(fg=ACCENT2 if busy else ACCENT)

    # ------------------------------------------------------------- actions #
    def pick_folder(self):
        path = filedialog.askdirectory(title="Choose a folder of documents")
        if path:
            self.source_var.set(path)

    def pick_files(self):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS))
        paths = filedialog.askopenfilenames(
            title="Choose documents",
            filetypes=[("Documents", exts), ("All files", "*.*")],
        )
        if paths:
            self.source_var.set(";".join(paths))

    def build_index(self):
        if self._busy:
            return
        raw = self.source_var.get().strip()
        if not raw:
            messagebox.showinfo(APP_NAME, "Choose a folder or files first.")
            return
        paths = [p for p in raw.split(";") if p.strip()]
        self._set_busy(True, "INDEXING...")

        def worker():
            try:
                idx = RagIndex()
                stats = idx.build(paths)
                self.index = idx
                os.makedirs(CONFIG_DIR, exist_ok=True)
                idx.save(INDEX_PATH)
                self.config_data["last_source"] = raw
                save_config(self.config_data)
                self.after(0, lambda: self._index_done(stats))
            except Exception:
                err = traceback.format_exc()
                self.after(0, lambda: self._index_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _index_done(self, stats: dict):
        self._set_busy(False)
        msg = f"INDEXED {stats['files']} FILE(S) / {stats['chunks']} PASSAGE(S)."
        if stats.get("errors"):
            msg += f" {len(stats['errors'])} SKIPPED."
        self.status_var.set(msg)
        if stats["chunks"] == 0:
            messagebox.showwarning(
                APP_NAME,
                "No readable text found. Supported types: "
                + ", ".join(sorted(SUPPORTED_EXTS)),
            )

    def _index_error(self, err: str):
        self._set_busy(False)
        self.status_var.set("INDEXING FAILED.")
        messagebox.showerror(APP_NAME, f"Indexing failed:\n\n{err}")

    def _try_load_saved_index(self):
        if os.path.exists(INDEX_PATH):
            try:
                self.index = RagIndex.load(INDEX_PATH)
                self.status_var.set(
                    f"LOADED INDEX: {len(self.index.sources)} FILE(S) / "
                    f"{len(self.index.chunks)} PASSAGE(S). ASK A QUESTION."
                )
            except Exception:
                self.index = None

    def ask(self):
        if self._busy:
            return
        query = self.query_var.get().strip()
        if not query:
            return
        if not self.index or not self.index.ready:
            messagebox.showinfo(APP_NAME, "Build an index first.")
            return

        self._set_busy(True, "SEARCHING...")
        top_k = max(1, int(self.topk_var.get()))
        llm_cfg = self.config_data.get("llm") or {}

        def worker():
            try:
                ans, hits = answer_query(query, self.index, top_k=top_k, llm_config=llm_cfg)
                self.after(0, lambda: self._show_results(ans, hits))
            except Exception:
                err = traceback.format_exc()
                self.after(0, lambda: self._ask_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _show_results(self, answer: str, hits):
        self._set_busy(False)
        self.status_var.set(f"{len(hits)} PASSAGE(S) RETRIEVED.")
        self._set_text(self.answer_box, answer)

        self.sources_box.configure(state="normal")
        self.sources_box.delete("1.0", "end")
        if not hits:
            self.sources_box.insert("end", "NO MATCHING PASSAGES.")
        for i, hit in enumerate(hits, 1):
            self.sources_box.insert("end", f" [{i}] {os.path.basename(hit.source)} ", "src")
            self.sources_box.insert("end", f"  SCORE {hit.score:.3f}\n", "score")
            self.sources_box.insert("end", f"{hit.source}\n", "path")
            self.sources_box.insert("end", hit.text.strip() + "\n\n")
        self.sources_box.configure(state="disabled")

    def _ask_error(self, err: str):
        self._set_busy(False)
        self.status_var.set("SEARCH FAILED.")
        messagebox.showerror(APP_NAME, f"Search failed:\n\n{err}")

    # ------------------------------------------------------------ settings #
    def open_settings(self):
        win = tk.Toplevel(self)
        win.title("SETTINGS")
        win.geometry("480x330")
        win.configure(bg=BG)
        win.transient(self)
        win.grab_set()

        llm = self.config_data.get("llm") or {}
        block_label(win, "// LLM SYNTHESIS (OPTIONAL)", bg=INK, fg=ACCENT, font=MONO_B).pack(
            fill="x", padx=16, pady=(16, 6)
        )
        tk.Label(
            win,
            text="NONE = FULLY OFFLINE (ANSWERS EXTRACTED FROM YOUR DOCS).\n"
            "GROQ = FREE, FAST AI ANSWERS. GET A KEY AT console.groq.com\n"
            "       (CREATE API KEY), PASTE IT BELOW, PICK PROVIDER = GROQ.",
            bg=BG, fg=INK, font=MONO_SM, justify="left", anchor="w",
        ).pack(fill="x", padx=16)

        grid = tk.Frame(win, bg=BG)
        grid.pack(fill="x", padx=16, pady=14)

        tk.Label(grid, text="PROVIDER", bg=BG, fg=INK, font=MONO_B).grid(row=0, column=0, sticky="w", pady=6)
        provider_var = tk.StringVar(value=llm.get("provider", "none") or "none")
        prov = tk.OptionMenu(grid, provider_var, "none", "groq", "openai", "anthropic")
        prov.configure(font=MONO_B, bg=PANEL, fg=INK, bd=0, highlightthickness=3,
                       highlightbackground=INK, activebackground=ACCENT, relief="flat", width=12)
        prov.grid(row=0, column=1, sticky="w", padx=10)

        tk.Label(grid, text="API KEY", bg=BG, fg=INK, font=MONO_B).grid(row=1, column=0, sticky="w", pady=6)
        key_var = tk.StringVar(value=llm.get("api_key", ""))
        ke = tk.Entry(grid, textvariable=key_var, show="*", font=MONO, bg=PANEL, fg=INK,
                      bd=0, highlightthickness=3, highlightbackground=INK, relief="flat", width=30)
        ke.grid(row=1, column=1, sticky="w", padx=10)

        tk.Label(grid, text="MODEL", bg=BG, fg=INK, font=MONO_B).grid(row=2, column=0, sticky="w", pady=6)
        model_var = tk.StringVar(value=llm.get("model", ""))
        me = tk.Entry(grid, textvariable=model_var, font=MONO, bg=PANEL, fg=INK,
                      bd=0, highlightthickness=3, highlightbackground=INK, relief="flat", width=30)
        me.grid(row=2, column=1, sticky="w", padx=10)
        tk.Label(grid, text="(BLANK = DEFAULT)", bg=BG, fg=MUTE, font=MONO_SM).grid(
            row=3, column=1, sticky="w", padx=10
        )

        def save_and_close():
            provider = provider_var.get()
            self.config_data["llm"] = {
                "provider": "" if provider == "none" else provider,
                "api_key": key_var.get().strip(),
                "model": model_var.get().strip(),
            }
            save_config(self.config_data)
            win.destroy()

        btns = tk.Frame(win, bg=BG)
        btns.pack(fill="x", padx=16, pady=10, side="bottom")
        hard_button(btns, "SAVE", save_and_close, bg=ACCENT).pack(side="right")
        hard_button(btns, "CANCEL", win.destroy).pack(side="right", padx=8)


def main():
    enable_dpi_awareness()
    App().mainloop()


if __name__ == "__main__":
    main()
