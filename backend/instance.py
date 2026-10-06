"""One Jarvis at a time: two would fight over the microphone and the HUD port."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


class InstanceLock:
    """An exclusive lock on a file holding the owner's pid, released when the process ends
    (even on a crash: the kernel drops flock locks with the file descriptor)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> bool:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd
        return True

    def owner(self) -> int | None:
        """The pid of the running instance, if it wrote one."""
        try:
            return int(self.path.read_text().strip())
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
