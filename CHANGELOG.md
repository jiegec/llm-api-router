# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-07

### Added

- Initial release of LLM API Router
- Separate OpenAI (`/openai/chat/completions`) and Anthropic (`/anthropic/v1/messages`) endpoints
- Native API format passthrough (no translation between formats)
- Streaming support for both endpoints with transparent chunk passthrough
- Priority-based multi-provider routing with automatic fallback
- Rate limit detection and cooldown mechanism for failed backends
- Web dashboard (`/web`) for real-time status monitoring with animated auto-refresh
- Analytics API endpoints (`/analytics/requests`, `/analytics/tokens`, `/analytics/latency`, `/analytics/summary`)
- Prometheus metrics endpoint (`/metrics`)
- Health check (`/health`) and status (`/status`) endpoints
- CSV logging for request statistics (`logs/request_stats.csv`)
- Structured JSON logging with per-request details
- Token usage tracking (input, output, cached)
- Streaming and non-streaming request statistics
- Rate limit statistics tracking with visualization
- Support for model name mapping via configuration
- CLI with `init`, `check`, and `serve` commands
- Support for `/anthropic/v1/messages/count_tokens` endpoint
- Prompt caching support with cached token tracking
- Support for OpenAI reasoning models (o1, o3, etc.) with `reasoning_content` in streaming responses
- SQL injection protection for analytics queries via prepared statements
- Complete time series generation for analytics charts (missing intervals show as zeros/NULL)

[Unreleased]: https://github.com/jiegec/llm-api-router/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jiegec/llm-api-router/releases/tag/v0.1.0
