"""Analytics module for querying request statistics using DuckDB."""

from pathlib import Path
from typing import Any

import duckdb


class AnalyticsQuery:
    """Analytics query engine using DuckDB."""

    def __init__(self, csv_path: str | Path = "logs/request_stats.csv"):
        """Initialize the analytics query engine.

        Args:
            csv_path: Path to the CSV file containing request statistics
        """
        self.csv_path = Path(csv_path)

    def _execute_query(self, query: str) -> list[dict[str, Any]]:
        """Execute a DuckDB query and return results as a list of dicts.

        Args:
            query: SQL query to execute

        Returns:
            List of dictionaries representing query results
        """
        if not self.csv_path.exists():
            return []

        con = duckdb.connect(":memory:")
        try:
            result = con.execute(query).fetchall()
            columns = [desc[0] for desc in con.description]
            return [dict(zip(columns, row, strict=True)) for row in result]
        finally:
            con.close()

    def get_requests_over_time(
        self,
        interval: str = "hour",
        hours: int = 24,
        provider_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get request count aggregated over time intervals.

        Args:
            interval: Time bucket size - 'minute', 'hour', or 'day'
            hours: Number of hours to look back (default: 24)
            provider_type: Filter by provider type ('openai', 'anthropic', or None for all)

        Returns:
            List of dicts with 'timestamp' and 'count' keys
        """
        # Map interval to DuckDB date_trunc part
        truncation = interval  # minute, hour, day work directly with date_trunc

        # Build query with time filter and optional provider filter
        provider_filter = (
            f"AND provider_type = '{provider_type}'" if provider_type else ""
        )

        query = f"""
            SELECT
                date_trunc('{truncation}', timestamp) AS timestamp,
                COUNT(*) AS count
            FROM read_csv_auto('{self.csv_path}', header=True)
            WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
              {provider_filter}
            GROUP BY timestamp
            ORDER BY timestamp
        """

        return self._execute_query(query)

    def get_tokens_over_time(
        self,
        interval: str = "hour",
        hours: int = 24,
        provider_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get token count aggregated over time intervals.

        Args:
            interval: Time bucket size - 'minute', 'hour', or 'day'
            hours: Number of hours to look back (default: 24)
            provider_type: Filter by provider type ('openai', 'anthropic', or None for all)

        Returns:
            List of dicts with 'timestamp', 'input_tokens', 'output_tokens', 'cached_tokens', 'total_tokens' keys
        """
        # Map interval to DuckDB date_trunc part
        truncation = interval  # minute, hour, day work directly with date_trunc

        # Build query with time filter and optional provider filter
        provider_filter = (
            f"AND provider_type = '{provider_type}'" if provider_type else ""
        )

        query = f"""
            SELECT
                date_trunc('{truncation}', timestamp) AS timestamp,
                SUM(input_tokens) AS input_tokens,
                SUM(output_tokens) AS output_tokens,
                SUM(cached_tokens) AS cached_tokens,
                SUM(input_tokens + output_tokens) AS total_tokens
            FROM read_csv_auto('{self.csv_path}', header=True)
            WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
              {provider_filter}
            GROUP BY timestamp
            ORDER BY timestamp
        """

        return self._execute_query(query)

    def get_latency_over_time(
        self,
        interval: str = "hour",
        hours: int = 24,
        provider_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get average latency aggregated over time intervals.

        Args:
            interval: Time bucket size - 'minute', 'hour', or 'day'
            hours: Number of hours to look back (default: 24)
            provider_type: Filter by provider type ('openai', 'anthropic', or None for all)

        Returns:
            List of dicts with 'timestamp', 'avg_latency_ms', 'p50_latency_ms', 'p95_latency_ms', 'p99_latency_ms' keys
        """
        # Map interval to DuckDB date_trunc part
        truncation = interval  # minute, hour, day work directly with date_trunc

        # Build query with time filter and optional provider filter
        provider_filter = (
            f"AND provider_type = '{provider_type}'" if provider_type else ""
        )

        query = f"""
            SELECT
                date_trunc('{truncation}', timestamp) AS timestamp,
                AVG(latency_ms) AS avg_latency_ms,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_latency_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_latency_ms
            FROM read_csv_auto('{self.csv_path}', header=True)
            WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
              {provider_filter}
            GROUP BY timestamp
            ORDER BY timestamp
        """

        return self._execute_query(query)

    def get_provider_summary(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get summary statistics by provider.

        Args:
            hours: Number of hours to look back (default: 24)

        Returns:
            List of dicts with provider stats
        """
        query = f"""
            SELECT
                provider_type,
                provider_name,
                COUNT(*) AS request_count,
                SUM(input_tokens + output_tokens) AS total_tokens,
                SUM(input_tokens) AS input_tokens,
                SUM(output_tokens) AS output_tokens,
                SUM(cached_tokens) AS cached_tokens,
                AVG(latency_ms) AS avg_latency_ms,
                SUM(CASE WHEN is_streaming = 'true' THEN 1 ELSE 0 END) AS streaming_count,
                SUM(CASE WHEN is_streaming = 'false' THEN 1 ELSE 0 END) AS non_streaming_count
            FROM read_csv_auto('{self.csv_path}', header=True)
            WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
            GROUP BY provider_type, provider_name
            ORDER BY request_count DESC
        """

        return self._execute_query(query)

    def get_available_time_range(self) -> dict[str, str] | None:
        """Get the available time range of data in the CSV.

        Returns:
            Dict with 'min_timestamp' and 'max_timestamp' keys, or None if no data
        """
        if not self.csv_path.exists():
            return None

        query = f"""
            SELECT
                MIN(timestamp) AS min_timestamp,
                MAX(timestamp) AS max_timestamp
            FROM read_csv_auto('{self.csv_path}', header=True)
        """

        results = self._execute_query(query)
        if results and results[0]["min_timestamp"]:
            return results[0]
        return None
