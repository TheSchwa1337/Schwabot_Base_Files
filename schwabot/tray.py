from __future__ import annotations

"""Cross-platform system-tray integration for Schwabot.

Requirements
~~~~~~~~~~~~
• pystray >= 0.19.4
• pillow >= 9.0

Install them with::

    pip install pystray pillow

The tray icon can be launched with::

    python -m schwabot.launch tray
"""

from pathlib import Path
import subprocess
import sys
import threading

try:
    import pystray  # type: ignore
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "pystray & pillow are required for the tray icon → `pip install pystray pillow`"
    ) from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).with_suffix("").parent / "assets"
ICON_PATH: Path = ASSETS_DIR / "schwabot.png"

if not ICON_PATH.exists():
    # Fallback: generate a 64×64 empty image so pystray still works.
    ICON_PATH.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64), (255, 128, 0, 255)).save(ICON_PATH)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def _open_gui(_: pystray.Icon, __: pystray.MenuItem) -> None:  # noqa: D401
    """Spawn the GUI process in a platform-independent way."""
    subprocess.Popen([sys.executable, "-m", "schwabot.launch", "gui"], close_fds=True)


def _quit(icon: pystray.Icon, _: pystray.MenuItem) -> None:  # noqa: D401
    """Terminate the tray application cleanly."""
    icon.stop()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_tray() -> None:
    """Start the Schwabot tray icon in a dedicated daemon thread."""
    image = Image.open(ICON_PATH)
    menu = pystray.Menu(
        pystray.MenuItem("Open Schwabot", _open_gui),
        pystray.MenuItem("Quit", _quit),
    )
    icon = pystray.Icon("Schwabot", image, "Schwabot", menu)

    # Run in its own thread so the caller regains control immediately.
    threading.Thread(target=icon.run, daemon=True).start()
