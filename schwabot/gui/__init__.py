from __future__ import annotations

"""Stub GUI layer for Schwabot.

In the future this could be a fully-fledged Qt / Tkinter / web-frontend.
For now it simply prints a message so that the "schwabot gui" command does
not crash.
"""

import subprocess
import sys
import tkinter as tk
from tkinter import messagebox


class SchwabotGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Schwabot Launcher")
        self.geometry("400x260")
        self.resizable(False, False)

        tk.Label(self, text="Schwabot", font=("Segoe UI", 24, "bold")).pack(pady=10)
        tk.Label(self, text="Choose a subsystem to launch:", font=("Segoe UI", 10)).pack(
            pady=(0, 15)
        )

        btn_frame = tk.Frame(self)
        btn_frame.pack()

        tk.Button(btn_frame, text="QSC Engine", width=15, command=self._launch_qsc).grid(
            row=0, column=0, padx=5, pady=5
        )
        tk.Button(btn_frame, text="Immune Engine", width=15, command=self._launch_immune).grid(
            row=0, column=1, padx=5, pady=5
        )
        tk.Button(btn_frame, text="Tensor Engine", width=15, command=self._launch_tensor).grid(
            row=1, column=0, padx=5, pady=5
        )
        tk.Button(btn_frame, text="Update", width=15, command=self._update).grid(
            row=1, column=1, padx=5, pady=5
        )

        tk.Button(self, text="Exit", command=self.destroy).pack(pady=15)

    # ---------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------

    def _spawn(self, args: list[str]) -> None:
        try:
            subprocess.Popen([sys.executable, "-m", "schwabot.launch", *args])
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Launch Error", str(exc))

    def _launch_qsc(self) -> None:
        self._spawn(["cli", "qsc", "start"])

    def _launch_immune(self) -> None:
        self._spawn(["cli", "immune", "start"])

    def _launch_tensor(self) -> None:
        self._spawn(["cli", "tensor", "start"])

    def _update(self) -> None:
        self._spawn(["cli", "update"])


def launch() -> None:  # noqa: D401
    gui = SchwabotGUI()
    gui.mainloop()
