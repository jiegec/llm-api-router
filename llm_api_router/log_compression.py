"""Log compression for LLM API Router.

Compresses log files that are at least 7 days old using zstd.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import zstandard as zstd

from .logging import get_logger

COMPRESS_AGE_DAYS = 7


def compress_old_logs(log_dir: str | Path = "logs") -> list[str]:
    """Compress log files that are at least 7 days old with zstd.

    Only compresses .jsonl and .csv files. Skips files that are already
    compressed (ending with .zst) or currently being written to.

    Args:
        log_dir: Directory containing log files.

    Returns:
        List of compressed file paths.
    """
    log_path = Path(log_dir)
    if not log_path.is_dir():
        return []

    logger = get_logger(str(log_dir))
    cutoff = datetime.now(timezone.utc) - timedelta(days=COMPRESS_AGE_DAYS)
    compressed_files: list[str] = []

    for log_file in sorted(log_path.iterdir()):
        if not log_file.is_file():
            continue
        if log_file.suffix == ".zst":
            continue
        if log_file.suffix != ".jsonl":
            continue

        try:
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue

        if mtime < cutoff:
            zst_path = log_file.with_suffix(log_file.suffix + ".zst")
            if zst_path.exists():
                continue

            try:
                original_size = log_file.stat().st_size
                _compress_file(log_file, zst_path)
                compressed_size = zst_path.stat().st_size
                log_file.unlink()
                compressed_files.append(str(zst_path))
                ratio = (
                    (1 - compressed_size / original_size) * 100 if original_size else 0
                )
                logger.logger.info(
                    f"Compressed old log: {log_file} -> {zst_path} "
                    f"({original_size} -> {compressed_size} bytes, {ratio:.1f}% reduction)"
                )
            except Exception as e:
                logger.logger.warning(f"Failed to compress {log_file}: {e}")

    return compressed_files


def _compress_file(src: Path, dst: Path) -> None:
    """Compress a single file using zstd at default level.

    The compressed file is written to a temporary file first, then
    atomically renamed to the destination to avoid partial files.

    Args:
        src: Source file path.
        dst: Destination .zst file path.
    """
    compressor = zstd.ZstdCompressor()
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")

    try:
        with open(src, "rb") as f_in:
            data = f_in.read()

        compressed = compressor.compress(data)

        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        try:
            os.write(fd, compressed)
        finally:
            os.close(fd)

        os.rename(str(tmp_path), str(dst))
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
