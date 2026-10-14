"""Keep Earth-Bench MCP stdout as newline-delimited JSON only.

``mcp.server.stdio.stdio_server`` writes each JSON-RPC message through a new
``TextIOWrapper`` on ``sys.stdout.buffer``, while the kits ``print()`` through
the original wrapper. Those two wrappers flush independently, so a diagnostic
and the tool result become one line. The client then cannot parse the
response (``Failed to parse JSONRPC message from server``) and, because
``ClientSession``'s read timeout defaults to ``None``, waits forever.

https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/stdio.py
https://docs.python.org/3/library/io.html#io.TextIOWrapper

This process is the kit's entry point. It points fd 1 at stderr (so ``print``
and C libraries such as GDAL leave the protocol pipe) and passes the original
stdout into ``stdio_server``.

``--stderr FILE`` before the kit path appends the kit's stderr (FastMCP's
banner, warnings, and tool tracebacks) to ``FILE`` instead of the client's
stderr, where a notebook shows each one as a red error.
"""

from __future__ import annotations

import os
import runpy
import sys
from typing import Any, BinaryIO


def redirect_protocol_stdout() -> BinaryIO:
    """Point fd 1 at stderr and return the original stdout as an unbuffered binary file."""
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    # closefd is already false on the interpreter's stdout, so closing the old
    # wrapper does not close fd 1. Replace it so print() uses the redirected fd.
    sys.stdout.close()
    sys.stdout = os.fdopen(1, "w", encoding="utf-8", buffering=1, closefd=False)
    return os.fdopen(protocol_fd, "wb", buffering=0)


def patch_stdio_server(protocol: BinaryIO) -> None:
    """Send MCP JSON-RPC to ``protocol`` instead of ``sys.stdout.buffer``."""
    import anyio
    from io import TextIOWrapper

    import mcp.server.stdio as mcp_stdio

    original = mcp_stdio.stdio_server

    def stdio_server(stdin: Any = None, stdout: Any = None) -> Any:
        if stdout is None:
            stdout = anyio.wrap_file(
                TextIOWrapper(protocol, encoding="utf-8", write_through=True)
            )
        return original(stdin=stdin, stdout=stdout)

    mcp_stdio.stdio_server = stdio_server


def redirect_stderr(path: str) -> None:
    """Append fd 2 to ``path``. Runs before fd 1 is pointed at stderr, so both land there."""
    sys.stderr.flush()
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 2)
    os.close(log_fd)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv if argv is None else argv)
    if len(args) >= 3 and args[1] == "--stderr":
        redirect_stderr(args[2])
        args = [args[0], *args[3:]]
    if len(args) < 2:
        raise SystemExit(
            f"usage: {args[0] if args else 'stdio_guard.py'} [--stderr FILE] KIT.py [args...]"
        )
    protocol = redirect_protocol_stdout()
    patch_stdio_server(protocol)
    sys.argv = args[1:]
    runpy.run_path(args[1], run_name="__main__")


if __name__ == "__main__":
    main()
