"""Analytics module for querying request statistics using DuckDB."""

import logging
from pathlib import Path
from typing import Any

import duckdb

logger = logging.getLogger("llm_api_router.analytics")


class AnalyticsQuery:
    """Analytics query engine using DuckDB."""

    # Valid values for enum-like parameters (those that can't be parameterized)
    VALID_INTERVALS = {"minute", "hour", "day"}

    def __init__(self, csv_path: str | Path = "logs/request_stats.csv"):
        """Initialize the analytics query engine.

        Args:
            csv_path: Path to the CSV file containing request statistics
        """
        self.csv_path = Path(csv_path).resolve()

    def _validate_interval(self, interval: str) -> str:
        """Validate interval parameter.

        Note: date_trunc() requires a literal string, so we validate
        via whitelist rather than using a parameter.

        Args:
            interval: The interval to validate

        Returns:
            The validated interval

        Raises:
            ValueError: If interval is not valid
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval '{interval}'. Must be one of: {self.VALID_INTERVALS}"
            )
        return interval

    def _execute_query(
        self, query: str, parameters: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a DuckDB query with optional parameters.

        Args:
            query: SQL query with ? placeholders for parameters
            parameters: Optional list of parameters for prepared statement

        Returns:
            List of dictionaries representing query results
        """
        # Log SQL query to stderr
        if parameters:
            logger.info(f"SQL: {query} | Parameters: {parameters}")
        else:
            logger.info(f"SQL: {query}")

        if not self.csv_path.exists():
            return []

        con = duckdb.connect(":memory:")
        try:
            if parameters:
                result = con.execute(query, parameters=parameters).fetchall()
            else:
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
        # Validate interval (can't be parameterized in date_trunc)
        interval = self._validate_interval(interval)

        if provider_type == "":
            provider_type = None

        # Determine the step for generating time series
        if interval == "minute":
            step = "INTERVAL 1 minute"
        elif interval == "hour":
            step = "INTERVAL 1 hour"
        else:  # day
            step = "INTERVAL 1 day"

        # Use prepared statements for hours and provider_type
        # Generate complete time series and LEFT JOIN actual data
        # Note: range() stop is exclusive, so add one step to include current bucket
        query = f"""
            WITH time_series AS (
                SELECT date_trunc('{interval}', ts) AS timestamp
                FROM (
                    SELECT timestamp AS ts
                    FROM range(
                        date_trunc('{interval}', NOW() - INTERVAL {int(hours)} hours),
                        date_trunc('{interval}', NOW() + {step}),
                        {step}
                    ) AS t(timestamp)
                )
            ),
            actual_data AS (
                SELECT
                    date_trunc('{interval}', timestamp) AS timestamp,
                    COUNT(*) AS count
                FROM read_csv_auto('{self.csv_path}', header=True)
                WHERE timestamp >= NOW() - INTERVAL {int(hours)} hours
                  AND ($1 IS NULL OR provider_type = $1)
                GROUP BY date_trunc('{interval}', timestamp)
            )
            SELECT
                ts.timestamp,
                COALESCE(ad.count, 0) AS count
            FROM time_series ts
            LEFT JOIN actual_data ad ON ts.timestamp = ad.timestamp
            ORDER BY ts.timestamp
        """

        return self._execute_query(query, parameters=[provider_type])

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
        # Validate interval (can't be parameterized in date_trunc)
        interval = self._validate_interval(interval)

        if provider_type == "":
            provider_type = None

        # Determine the step for generating time series
        if interval == "minute":
            step = "INTERVAL 1 minute"
        elif interval == "hour":
            step = "INTERVAL 1 hour"
        else:  # day
            step = "INTERVAL 1 day"

        # Use prepared statements for hours and provider_type
        # Generate complete time series and LEFT JOIN actual data
        # Note: range() stop is exclusive, so add one step to include current bucket
        query = f"""
            WITH time_series AS (
                SELECT date_trunc('{interval}', ts) AS timestamp
                FROM (
                    SELECT timestamp AS ts
                    FROM range(
                        date_trunc('{interval}', NOW() - INTERVAL {int(hours)} hours),
                        date_trunc('{interval}', NOW() + {step}),
                        {step}
                    ) AS t(timestamp)
                )
            ),
            actual_data AS (
                SELECT
                    date_trunc('{interval}', timestamp) AS timestamp,
                    SUM(input_tokens) AS input_tokens,
                    SUM(output_tokens) AS output_tokens,
                    SUM(cached_tokens) AS cached_tokens,
                    SUM(input_tokens + output_tokens) AS total_tokens
                FROM read_csv_auto('{self.csv_path}', header=True)
                WHERE timestamp >= NOW() - INTERVAL {int(hours)} hours
                  AND ($1 IS NULL OR provider_type = $1)
                GROUP BY date_trunc('{interval}', timestamp)
            )
            SELECT
                ts.timestamp,
                COALESCE(ad.input_tokens, 0) AS input_tokens,
                COALESCE(ad.output_tokens, 0) AS output_tokens,
                COALESCE(ad.cached_tokens, 0) AS cached_tokens,
                COALESCE(ad.total_tokens, 0) AS total_tokens
            FROM time_series ts
            LEFT JOIN actual_data ad ON ts.timestamp = ad.timestamp
            ORDER BY ts.timestamp
        """

        return self._execute_query(query, parameters=[provider_type])

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
        # Validate interval (can't be parameterized in date_trunc)
        interval = self._validate_interval(interval)

        if provider_type == "":
            provider_type = None

        # Determine the step for generating time series
        if interval == "minute":
            step = "INTERVAL 1 minute"
        elif interval == "hour":
            step = "INTERVAL 1 hour"
        else:  # day
            step = "INTERVAL 1 day"

        # Use prepared statements for hours and provider_type
        # Generate complete time series and LEFT JOIN actual data
        # Note: range() stop is exclusive, so add one step to include current bucket
        query = f"""
            WITH time_series AS (
                SELECT date_trunc('{interval}', ts) AS timestamp
                FROM (
                    SELECT timestamp AS ts
                    FROM range(
                        date_trunc('{interval}', NOW() - INTERVAL {int(hours)} hours),
                        date_trunc('{interval}', NOW() + {step}),
                        {step}
                    ) AS t(timestamp)
                )
            ),
            actual_data AS (
                SELECT
                    date_trunc('{interval}', timestamp) AS timestamp,
                    AVG(latency_ms) AS avg_latency_ms,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_latency_ms,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_latency_ms
                FROM read_csv_auto('{self.csv_path}', header=True)
                WHERE timestamp >= NOW() - INTERVAL {int(hours)} hours
                  AND ($1 IS NULL OR provider_type = $1)
                GROUP BY date_trunc('{interval}', timestamp)
            )
            SELECT
                ts.timestamp,
                ad.avg_latency_ms,
                ad.p50_latency_ms,
                ad.p95_latency_ms,
                ad.p99_latency_ms
            FROM time_series ts
            LEFT JOIN actual_data ad ON ts.timestamp = ad.timestamp
            ORDER BY ts.timestamp
        """

        return self._execute_query(query, parameters=[provider_type])

    def get_provider_summary(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get summary statistics by provider.

        Args:
            hours: Number of hours to look back (default: 24)

        Returns:
            List of dicts with provider stats
        """
        # Use prepared statement for hours
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
            WHERE timestamp >= NOW() - INTERVAL {int(hours)} hours
            GROUP BY provider_type, provider_name
            ORDER BY request_count DESC
        """

        return self._execute_query(query, parameters=[])

    def get_available_time_range(self) -> dict[str, str] | None:
        """Get the available time range of data in the CSV.

        Returns:
            Dict with 'min_timestamp' and 'max_timestamp' keys, or None if no data
        """
        if not self.csv_path.exists():
            return None

        # No user input, no parameters needed
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
