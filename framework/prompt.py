"""
prompt.py – Robust ANSI status line at bottom of console (Windows CMD fixed).
Uses Colorama for VT support (no registry needed).
Integrated with tasks.py manager.
Call enable_prompt() BEFORE stream.run().
"""

"""
prompt.py – Bottom status line using 'blessed'.
Call enable_prompt() BEFORE stream.run().
"""
'''
import asyncio
import sys
from . import tasks

# --------------------------------------------------------------------------- #
# Blessed terminal
# --------------------------------------------------------------------------- #
from blessed import Terminal
_term = Terminal()

# --------------------------------------------------------------------------- #
# Global state (module-level)
# --------------------------------------------------------------------------- #
_caption: str = ">> "  # ← Initialized here
_lock = asyncio.Lock()
_redraw_event = asyncio.Event()

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def set_caption(text: str) -> None:
    """Change the status line text."""
    
    async def _set():
        global _caption  # ← Explicitly reference global
        async with _lock:
            if _caption != text:
                _caption = text
                _redraw_event.set()

    # Fire and forget
    asyncio.create_task(_set())

def get_caption() -> str:
    return _caption

# --------------------------------------------------------------------------- #
# Background task
# --------------------------------------------------------------------------- #
async def _status_line_task() -> None:
    print(_term.clear_eos, end="")        # Clear screen below
    print(_term.hide_cursor, end="")

    while True:
        await _redraw_event.wait()
        await asyncio.sleep(0.05)         # Debounce
        async with _lock:
            _redraw_event.clear()
            caption = _caption or "ready"

        with _term.location(0, _term.height - 1):
            print(_term.clear_eol + f"{caption}" + _term.clear_eol, end="")
        sys.stdout.flush()

# --------------------------------------------------------------------------- #
# Task registration
# --------------------------------------------------------------------------- #
_prompt_task_registered = False

def enable_prompt(initial_caption: str = "") -> None:
    global _prompt_task_registered, _caption
    if _prompt_task_registered:
        return
    _prompt_task_registered = True

    _caption = initial_caption
    tasks.registerTask("status_line", _status_line_task)

# --------------------------------------------------------------------------- #
# Shutdown
# --------------------------------------------------------------------------- #
def disable_prompt() -> None:
    tasks.cancelTask("status_line")
    print(_term.normal + _term.show_cursor, end="")
    sys.stdout.flush()

'''


'''
import asyncio
import sys
from typing import Callable
from . import tasks

# --------------------------------------------------------------------------- #
# Colorama: Enable ANSI in Windows CMD (no registry, no admin)
# --------------------------------------------------------------------------- #
try:
    from colorama import init, AnsiToWin32
    _use_colorama = True
    init(autoreset=False)  # Enable VT processing
    _stdout = AnsiToWin32(sys.stdout).stream
except ImportError:
    _use_colorama = False
    _stdout = sys.stdout
    print("Warning: pip install colorama for Windows CMD support.")

# --------------------------------------------------------------------------- #
# Global state
# --------------------------------------------------------------------------- #
_caption: str = ""
_lock = asyncio.Lock()
_redraw_event = asyncio.Event()

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def set_caption(text: str) -> None:
    """Change the status line text. Safe from any context."""
    global _caption
    async def _set():
        async with _lock:
            if _caption != text:
                _caption = text
                _redraw_event.set()
    asyncio.create_task(_set())

def get_caption() -> str:
    return _caption

# --------------------------------------------------------------------------- #
# Background redraw task
# --------------------------------------------------------------------------- #
async def _status_line_task() -> None:
    """Keep status line at bottom of screen."""
    # Hide cursor
    _stdout.write("\033[?25l")
    _stdout.flush()

    while True:
        await _redraw_event.wait()
        await asyncio.sleep(0.05)  # Debounce rapid updates
        async with _lock:
            _redraw_event.clear()
            caption = _caption or "ready"

        height = _get_terminal_height()
        if height < 1:
            height = 24

        # Save cursor, go to bottom, clear line, print, restore
        esc = "\033"
        _stdout.write(f"{esc}[s")                    # Save cursor
        _stdout.write(f"{esc}[{height};1H")          # Bottom-left
        _stdout.write(f"{esc}[2K")                   # Clear line
        _stdout.write(f"\r[ {caption} ]{esc}[K")     # Print + clear rest
        _stdout.write(f"{esc}[u")                    # Restore cursor
        _stdout.flush()

def _get_terminal_height() -> int:
    try:
        import shutil
        return shutil.get_terminal_size(fallback=(80, 24)).lines
    except:
        return 24

# --------------------------------------------------------------------------- #
# Task registration
# --------------------------------------------------------------------------- #
_prompt_task_registered = False

def enable_prompt(initial_caption: str = "") -> None:
    """Register the status line task. Call BEFORE stream.run()."""
    global _prompt_task_registered, _caption
    if _prompt_task_registered:
        return
    _prompt_task_registered = True

    _caption = initial_caption
    tasks.registerTask("status_line", _status_line_task)  # ← Queued for task manager

# --------------------------------------------------------------------------- #
# Graceful shutdown
# --------------------------------------------------------------------------- #
def disable_prompt() -> None:
    """Cancel task and restore cursor."""
    tasks.cancelTask("status_line")
    _stdout.write("\033[?25h")
    _stdout.flush()

'''


"""
prompt_cli.py
Always-on-bottom prompt that pulls its caption via a user-supplied callback.
No bank, no setCaption, no repaint headaches.
"""

'''
import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import HTML
from typing import Callable
from . import active
from . import tasks


# --------------------------------------------------------------------------- #

#  prompt task                                                                 #

# --------------------------------------------------------------------------- #
class LivePrompt:
    """
    Create-and-forget prompt task.
    caption_func: sync function -> str  (called on every redraw)
    """

    def __init__(self, caption_func: Callable[[], str]) -> None:
        self.session = PromptSession()
        self.caption_func = caption_func
        self._stop = asyncio.Event()
        self._last_caption = None  # Track last known caption for change detection

    async def run(self) -> None:
        """Run forever until `stop()` is called."""
        with patch_stdout(raw=True):
            asyncio.create_task(self._redraw_loop())  # Start background redraw poller
            while not self._stop.is_set():
                try:
                    line = await self.session.prompt_async(
                        message=lambda: HTML(self._color(self.caption_func(), "#bbbbbb"))
                    )
                except (EOFError, KeyboardInterrupt):
                    break
                await self._handle(line.strip())

    async def _redraw_loop(self) -> None:
        """Poll for caption changes and invalidate to redraw if changed."""
        while not self._stop.is_set():
            await asyncio.sleep(0.2)  # Poll interval (adjust if needed; 0.2s is responsive but low-overhead)
            current = self.caption_func()
            if current != self._last_caption:
                self._last_caption = current
                if self.session.app:
                    self.session.app.invalidate()

    def stop(self) -> None:
        self._stop.set()

    # ---------------------------- helpers -------------------------------- #
    @staticmethod
    def _color(text: str, color: str) -> str:
        return f'<b><style color="{color}">{text}</style></b> '
    # -------------------------- command router ---------------------------- #

    async def _handle(self, line: str) -> None:
        stream = active.timeframe.stream

        parts   = line.split(" ", 1)
        command = parts[0].lower()
        args    = parts[1] if len(parts) > 1 else ""

        if command in {"chart", "c"}:
            stream.createWindow(args)
        else:
            stream.event_callback(stream, "cli_command", (command, args), 2)


_prompt_started = False

def enable_prompt():
    global _prompt_started
    if _prompt_started:
        return
    _prompt_started = True

    prompt = LivePrompt(lambda: (active.timeframe.stream.caption + " > " if active.timeframe and hasattr(active.timeframe, 'stream') and hasattr(active.timeframe.stream, 'caption') else " > "))
    tasks.registerTask("live_prompt", prompt.run)
'''