import ctypes
import os
import select
import signal
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence


_PIDFD_OPEN_SYSCALLS = {"x86_64": 434, "aarch64": 434, "ppc64le": 434}


def open_process_handle(pid: int) -> int:
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    syscall_number = _PIDFD_OPEN_SYSCALLS.get(os.uname().machine)
    if syscall_number is None:
        raise OSError(f"pidfd_open is unavailable on {os.uname().machine}")
    libc = ctypes.CDLL(None, use_errno=True)
    handle = libc.syscall(syscall_number, pid, 0)
    if handle == -1:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    return handle


def pidfd_supported() -> bool:
    try:
        handle = open_process_handle(os.getpid())
    except OSError:
        return False
    os.close(handle)
    return True


def process_handle_is_running(handle: int) -> bool:
    poller = select.poll()
    poller.register(handle, select.POLLIN)
    return not poller.poll(0)


def assert_a3m_query(path: Path, expected_id: str, expected_sequence: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    header: str | None = None
    sequence_parts: list[str] = []
    for line in lines:
        if header is None:
            if not line or line.startswith("#"):
                continue
            assert line.startswith(">"), f"{path}: first A3M record has no header"
            header = line[1:]
            continue
        if line.startswith(">"):
            break
        if line and not line.startswith("#"):
            sequence_parts.append(line)

    assert header is not None, f"{path}: first A3M record is missing"
    assert header, f"{path}: first A3M record has an empty header"
    assert header.split(maxsplit=1)[0] == expected_id, (
        f"{path}: first A3M record identifier does not match {expected_id!r}"
    )
    query_sequence = "".join(sequence_parts)
    assert query_sequence, f"{path}: query sequence is empty"
    assert query_sequence == expected_sequence, (
        f"{path}: query sequence does not match input sequence for {expected_id!r}"
    )


def resolve_path(value: str, parent_cwd: Path) -> Path:
    path = Path(value).expanduser()
    return (parent_cwd / path).resolve() if not path.is_absolute() else path.resolve()


def resolve_executable(value: str, parent_cwd: Path, path: str | None = None) -> str:
    if os.sep not in value and (os.altsep is None or os.altsep not in value):
        resolved = shutil.which(value, path=os.environ.get("PATH") if path is None else path)
        return str(resolve_path(resolved, parent_cwd)) if resolved is not None else value
    return str(resolve_path(value, parent_cwd))


def configure_search_environment(
    child_env: MutableMapping[str, str], parent_cwd: Path
) -> str:
    executable = child_env.get("COLABFOLD_SEARCH", "colabfold_search")
    resolved = resolve_executable(executable, parent_cwd, path=child_env.get("PATH"))
    child_env["COLABFOLD_SEARCH"] = resolved
    return resolved


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    if os.name != "posix":
        process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, AttributeError):
        pass


def _reap_process(process: subprocess.Popen[str]) -> None:
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_with_timeout(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            _reap_process(process)
        error = subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr)
        error.pid = process.pid
        raise error from None
    except BaseException:
        _kill_process_group(process)
        _reap_process(process)
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def failure_details(output_dir: Path, tmp_dir: Path, stdout: str, stderr: str) -> str:
    try:
        published_log = output_dir / "run.log"
        candidates = [published_log] if published_log.is_file() else list(
            (tmp_dir / "cluster-msa-work").glob("standard-*/output-*/run.log")
        )
        run_log_path = max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None
        run_log = (
            run_log_path.read_text(encoding="utf-8", errors="replace")
            if run_log_path is not None
            else "<not available>"
        )
    except OSError as error:
        run_log_path = None
        run_log = f"<not available: {error}>"
    log_label = str(run_log_path) if run_log_path is not None else "run.log"
    return f"stdout:\n{stdout}\n\nstderr:\n{stderr}\n\n{log_label}:\n{run_log}"
