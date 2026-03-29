---
name: data-worker
description: Worker for data ingestion, text processing, and chunking operations
---

# Data Worker

Handles Wikipedia scraping, text processing, and heuristic chunking.

## When to Use This Skill

Use for features involving:
- Wikipedia API interaction and article scraping
- Text processing and cleaning
- Custom chunking algorithms with sentence boundary detection
- Data persistence (HDF5, JSON, etc.)

## Required Skills

None

## Work Procedure

1. **Write tests first (TDD)**
   - Create test file with pytest
   - Tests must fail before implementation
   - Cover edge cases: empty text, single sentence, exact boundary, overlapping chunks

2. **Implement the feature**
   - Use `wikipedia-api` library for Wikipedia access
   - Implement proper User-Agent headers
   - Add rate limiting (0.5s delay between requests)
   - For chunking: respect sentence boundaries, implement exact token counts

3. **Manual verification**
   - Run the script and verify output
   - Check that chunks respect sentence boundaries
   - Verify token counts are correct
   - Log sample chunks for review

4. **Run validators**
   - `python -m pytest tests/ -v`
   - Fix any failures

5. **Commit work**
   - Clear commit message describing what was implemented

## Example Handoff

```json
{
  "salientSummary": "Implemented Wikipedia scraper fetching top 10 Quantum Cryptography articles and custom chunking algorithm producing 500-token chunks with 50-token overlap while respecting sentence boundaries.",
  "whatWasImplemented": "Created wikipedia_scraper.py using wikipedia-api library with proper User-Agent and rate limiting. Fetched 10 articles with full text content. Implemented heuristic_chunker.py with sentence boundary detection using NLTK/nltk.sent_tokenize, producing chunks of exactly 500 tokens with 50-token overlap. Chunks never truncate mid-sentence.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "python -m pytest tests/test_chunker.py -v", "exitCode": 0, "observation": "6 tests passed including edge cases for empty text, single sentence, and boundary conditions"},
      {"command": "python -m pytest tests/test_scraper.py -v", "exitCode": 0, "observation": "3 tests passed for API interaction and rate limiting"},
      {"command": "python scripts/test_scraper_manual.py", "exitCode": 0, "observation": "Successfully fetched 10 articles, total text length: 45,230 characters, saved to data/raw_articles.json"}
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {"file": "tests/test_chunker.py", "cases": [
        {"name": "test_chunk_respects_sentence_boundaries", "verifies": "No chunk ends mid-sentence"},
        {"name": "test_chunk_size_500_tokens", "verifies": "Each chunk is at most 500 tokens"},
        {"name": "test_overlap_50_tokens", "verifies": "Consecutive chunks share 50 tokens"},
        {"name": "test_empty_text", "verifies": "Empty input returns empty list"},
        {"name": "test_single_sentence", "verifies": "Single sentence smaller than chunk size returns as one chunk"},
        {"name": "test_exact_boundary", "verifies": "Sentence ending exactly at 500 tokens is handled correctly"}
      ]},
      {"file": "tests/test_scraper.py", "cases": [
        {"name": "test_user_agent_present", "verifies": "Wikipedia client has proper User-Agent"},
        {"name": "test_rate_limiting", "verifies": "Requests have delay between them"},
        {"name": "test_fetch_articles_count", "verifies": "Returns exactly 10 articles"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Wikipedia API is unavailable or rate limiting is too aggressive
- Sentence boundary detection library unavailable
- Requirements for chunk size/overlap are contradictory
- Discovered that more articles are needed than initially planned
