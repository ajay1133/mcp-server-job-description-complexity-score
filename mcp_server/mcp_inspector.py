#!/usr/bin/env python3
"""
Launch the Model Context Protocol Inspector and spawn this project's MCP server.

This wrapper ensures the server starts using the same Python environment used to
invoke this command (e.g., via `uv run mcp-inspect-server`). It requires Node.js
and `npx` to be available on PATH.
"""
import subprocess
import sys
import shutil
import os


def main() -> int:
    # Determine how to launch npx reliably on Windows and POSIX
    npx_exe = shutil.which("npx") or shutil.which("npx.cmd") or shutil.which("npx.exe")
    if npx_exe is None:
        print("Error: `npx` not found on PATH. Please install Node.js (which includes npx).", file=sys.stderr)
        return 1

    # Use the current Python interpreter to run the server as a module
    python_exec = sys.executable
    server_module = "mcp_server.server"

    # Build command; on Windows prefer using the string form with shell=True for .cmd resolution
    inspector_pkg = "@modelcontextprotocol/inspector"
    base_args = f"{inspector_pkg} --server.command \"{python_exec}\" --server.args \"-m {server_module}\""

    # Pass through any additional args to the inspector
    extra = " " + " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""

    use_shell = os.name == "nt"
    if use_shell:
        # Quote npx path to handle spaces (e.g., Program Files)
        cmd = f'"{npx_exe}" {base_args}{extra}'
    else:
        cmd = [
            npx_exe,
            inspector_pkg,
            "--server.command",
            python_exec,
            "--server.args",
            f"-m {server_module}",
            *sys.argv[1:],
        ]

    try:
        return subprocess.call(cmd, shell=use_shell)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
