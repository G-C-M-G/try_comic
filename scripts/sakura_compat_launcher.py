"""Start the upstream Sakura server with its CUDA runtime preloaded.

Python 3.11 no longer searches PATH for transitive DLL dependencies in the
same way as older Python versions.  Preloading the bundled CUDA 12 runtime
lets llama-cpp-python use its own version-matched llama.dll.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import runpy
import sys


CUDA_RUNTIME_DLLS = (
    "cudart64_12.dll",
    "cublasLt64_12.dll",
    "cublas64_12.dll",
)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: sakura_compat_launcher.py <server.py> <dll-directory>")

    server_path = Path(sys.argv[1]).resolve()
    dll_directory = Path(sys.argv[2]).resolve()
    if not server_path.is_file():
        raise FileNotFoundError(server_path)
    if not dll_directory.is_dir():
        raise NotADirectoryError(dll_directory)

    os.environ.pop("LLAMA_CPP_LIB_PATH", None)
    os.environ["PATH"] = f"{dll_directory};{os.environ.get('PATH', '')}"
    dll_search_handle = os.add_dll_directory(str(dll_directory))
    loaded_libraries = [
        ctypes.WinDLL(str(dll_directory / name)) for name in CUDA_RUNTIME_DLLS
    ]

    # Keep both the search handle and loaded libraries alive for the server.
    _ = dll_search_handle, loaded_libraries
    runpy.run_path(str(server_path), run_name="__main__")


if __name__ == "__main__":
    main()
