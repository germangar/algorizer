
"""
prompt_cli.py
Always-on-bottom prompt that pulls its caption via a user-supplied callback.
No bank, no setCaption, no repaint headaches.
"""
import asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import HTML
from typing import Callable
from . import active


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