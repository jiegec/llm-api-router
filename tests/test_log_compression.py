"""Tests for log compression."""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llm_api_router.log_compression import compress_old_logs


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir()
    return d


def _set_mtime(path: Path, dt: datetime) -> None:
    timestamp = dt.timestamp()
    os.utime(path, (timestamp, timestamp))


class TestCompressOldLogs:
    def test_compresses_old_jsonl(self, log_dir: Path) -> None:
        f = log_dir / "router_old.jsonl"
        f.write_text('{"test": true}\n')
        _set_mtime(f, datetime.now(timezone.utc) - timedelta(days=10))

        result = compress_old_logs(log_dir)
        assert len(result) == 1
        assert result[0].endswith(".jsonl.zst")
        assert not f.exists()
        assert Path(result[0]).exists()

    def test_skips_csv_files(self, log_dir: Path) -> None:
        f = log_dir / "request_stats.csv"
        f.write_text("a,b\n1,2\n")
        _set_mtime(f, datetime.now(timezone.utc) - timedelta(days=8))

        result = compress_old_logs(log_dir)
        assert result == []
        assert f.exists()

    def test_skips_recent_files(self, log_dir: Path) -> None:
        f = log_dir / "router_recent.jsonl"
        f.write_text('{"test": true}\n')

        result = compress_old_logs(log_dir)
        assert result == []
        assert f.exists()

    def test_skips_non_log_files(self, log_dir: Path) -> None:
        f = log_dir / "readme.txt"
        f.write_text("hello")
        _set_mtime(f, datetime.now(timezone.utc) - timedelta(days=30))

        result = compress_old_logs(log_dir)
        assert result == []
        assert f.exists()

    def test_skips_already_compressed(self, log_dir: Path) -> None:
        f = log_dir / "router_old.jsonl.zst"
        f.write_bytes(b"\x28\xb5\x2f\xfd")
        _set_mtime(f, datetime.now(timezone.utc) - timedelta(days=30))

        result = compress_old_logs(log_dir)
        assert result == []

    def test_skips_existing_zst(self, log_dir: Path) -> None:
        src = log_dir / "router_old.jsonl"
        src.write_text('{"test": true}\n')
        _set_mtime(src, datetime.now(timezone.utc) - timedelta(days=10))

        zst = log_dir / "router_old.jsonl.zst"
        zst.write_bytes(b"\x28\xb5\x2f\xfd")

        result = compress_old_logs(log_dir)
        assert result == []
        assert src.exists()

    def test_compressed_file_is_valid_zstd(self, log_dir: Path) -> None:
        import zstandard as zstd

        content = '{"key": "value"}\n{"another": "line"}\n'
        f = log_dir / "router_old.jsonl"
        f.write_text(content)
        _set_mtime(f, datetime.now(timezone.utc) - timedelta(days=10))

        result = compress_old_logs(log_dir)
        assert len(result) == 1

        decompressor = zstd.ZstdDecompressor()
        with open(result[0], "rb") as f_in:
            decompressed = decompressor.decompress(f_in.read())
        assert decompressed.decode("utf-8") == content

    def test_nonexistent_log_dir(self, tmp_path: Path) -> None:
        result = compress_old_logs(tmp_path / "nonexistent")
        assert result == []

    def test_empty_log_dir(self, log_dir: Path) -> None:
        result = compress_old_logs(log_dir)
        assert result == []

    def test_mixed_files(self, log_dir: Path) -> None:
        old_jsonl = log_dir / "router_old.jsonl"
        old_jsonl.write_text("old\n")
        _set_mtime(old_jsonl, datetime.now(timezone.utc) - timedelta(days=10))

        recent_jsonl = log_dir / "router_recent.jsonl"
        recent_jsonl.write_text("recent\n")

        txt_file = log_dir / "notes.txt"
        txt_file.write_text("notes\n")
        _set_mtime(txt_file, datetime.now(timezone.utc) - timedelta(days=20))

        result = compress_old_logs(log_dir)
        assert len(result) == 1
        assert Path(result[0]).name == "router_old.jsonl.zst"
        assert recent_jsonl.exists()
        assert txt_file.exists()
