-- Prefect Docs MCP Performance Dashboard
-- Tested queries for Logfire (Apache DataFusion SQL)

-- ============================================================================
-- Request Rate & Latency Trends (24 hours)
-- ============================================================================
SELECT
  date_trunc('hour', start_timestamp) as hour,
  COUNT(*) as requests,
  ROUND(AVG(duration)::numeric, 3) as avg_sec,
  ROUND(approx_percentile_cont(duration, 0.95)::numeric, 3) as p95_sec,
  ROUND(approx_percentile_cont(duration, 0.99)::numeric, 3) as p99_sec
FROM records
WHERE span_name = 'search_prefect'
  AND start_timestamp > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;

-- ============================================================================
-- Most Common Queries (7 days)
-- ============================================================================
SELECT
  attributes->>'query' as query,
  COUNT(*) as frequency,
  ROUND(AVG((attributes->>'score_avg')::float)::numeric, 3) as avg_score,
  ROUND(AVG(duration)::numeric, 3) as avg_latency_sec
FROM records
WHERE span_name = 'search_prefect'
  AND start_timestamp > NOW() - INTERVAL '7 days'
GROUP BY attributes->>'query'
ORDER BY frequency DESC
LIMIT 25;

-- ============================================================================
-- Low Quality Queries (score < 0.5, last 7 days)
-- ============================================================================
SELECT
  attributes->>'query' as query,
  ROUND(AVG((attributes->>'score_avg')::float)::numeric, 3) as avg_score,
  COUNT(*) as frequency
FROM records
WHERE span_name = 'search_prefect'
  AND start_timestamp > NOW() - INTERVAL '7 days'
  AND (attributes->>'score_avg')::float < 0.5
GROUP BY attributes->>'query'
ORDER BY frequency DESC
LIMIT 20;

-- ============================================================================
-- Vector Query Performance
-- ============================================================================
SELECT
  date_trunc('hour', start_timestamp) as hour,
  COUNT(*) as queries,
  ROUND(AVG(duration)::numeric, 3) as avg_sec,
  ROUND(MAX(duration)::numeric, 3) as max_sec
FROM records
WHERE span_name = 'vector_query'
  AND start_timestamp > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;

-- ============================================================================
-- Empty Results Rate (24 hours)
-- ============================================================================
SELECT
  date_trunc('hour', start_timestamp) as hour,
  COUNT(*) FILTER (WHERE (attributes->>'result_count')::int = 0) as empty_results,
  COUNT(*) as total_queries,
  ROUND(100.0 * COUNT(*) FILTER (WHERE (attributes->>'result_count')::int = 0) / COUNT(*), 2) as empty_pct
FROM records
WHERE span_name = 'search_prefect'
  AND start_timestamp > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;

-- ============================================================================
-- Error Summary (7 days)
-- ============================================================================
SELECT
  attributes->>'error_type' as error_type,
  COUNT(*) as count,
  MIN(start_timestamp) as first_seen,
  MAX(start_timestamp) as last_seen
FROM records
WHERE span_name = 'search_prefect'
  AND attributes->>'error_type' IS NOT NULL
  AND start_timestamp > NOW() - INTERVAL '7 days'
GROUP BY attributes->>'error_type'
ORDER BY count DESC;

-- ============================================================================
-- Slow Queries (>2 seconds, last 24h)
-- ============================================================================
SELECT
  start_timestamp,
  attributes->>'query' as query,
  ROUND(duration::numeric, 3) as latency_sec,
  (attributes->>'result_count')::int as results
FROM records
WHERE span_name = 'search_prefect'
  AND duration > 2
  AND start_timestamp > NOW() - INTERVAL '24 hours'
ORDER BY duration DESC
LIMIT 50;

-- ============================================================================
-- Health Check (last 5 minutes)
-- ============================================================================
SELECT
  COUNT(*) as requests,
  ROUND(AVG(duration)::numeric, 3) as avg_latency_sec,
  COUNT(*) FILTER (WHERE attributes->>'error_type' IS NOT NULL) as errors,
  ROUND(100.0 * COUNT(*) FILTER (WHERE (attributes->>'result_count')::int > 0) / COUNT(*), 2) as success_pct
FROM records
WHERE span_name = 'search_prefect'
  AND start_timestamp > NOW() - INTERVAL '5 minutes';
