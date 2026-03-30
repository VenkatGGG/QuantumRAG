# Chunking Contract: Option A

## Overview

This document specifies **Option A** of the chunking contract for the RAG application's text processing pipeline. This contract balances sentence integrity with practical chunking constraints.

## Core Principles

### 1. Sentence Boundary Respect (Primary)
**Rule:** Chunk boundaries NEVER cut sentences mid-way.

- A chunk must end at a complete sentence boundary (after `.`, `!`, or `?`)
- No sentence may be split across chunks
- This is the highest-priority constraint

### 2. Maximum Chunk Size
**Rule:** Each chunk contains at most 500 tokens.

- Measured using the model's tokenizer (`sentence-transformers/all-MiniLM-L6-v2`)
- If a single sentence exceeds 500 tokens, it is truncated (emergency fallback)
- This is a hard ceiling, not a target

### 3. Sentence-Aligned Overlap (Target-Based)
**Rule:** Consecutive chunks share overlap that is **sentence-aligned and closest to 50 tokens**.

- **IMPORTANT:** 50 tokens is a TARGET, not a guarantee
- Overlap consists of complete sentences from the previous chunk
- The algorithm selects sentences that get as close as possible to 50 tokens
- Actual overlap may vary (typically 30-70 tokens) depending on sentence lengths
- The overlap is chosen to minimize `|actual_tokens - 50|`

## Algorithm

### Step 1: Wikipedia Segmentation
Before chunking, text is segmented at natural boundaries:

1. **Headings** - Lines matching heading patterns (e.g., `==Section==` in Wikipedia markup)
2. **Blank lines** - Empty lines separate logical sections
3. **Formulas** - LaTeX/math blocks are treated as atomic units
4. **Lists** - Bullet and numbered list items are treated as sentences

### Step 2: Sentence Tokenization
- Text is split into sentences using regex-based boundary detection
- Pattern: `(?<=[.!?])\s+` (splits after punctuation followed by whitespace)
- Sentences are trimmed and empty ones filtered

### Step 3: Chunk Construction
```
For each chunk:
1. If overlap exists from previous chunk:
   - Start with overlap sentences (sentence-aligned, ~50 tokens target)
   - Calculate remaining budget: 500 - overlap_tokens
2. Otherwise (first chunk):
   - Full 500 tokens available

3. Add new sentences until adding another would exceed budget
4. Verify chunk size <= 500 (truncate if necessary)
5. Determine overlap for next chunk:
   - Walk backwards through this chunk's new sentences
   - Include sentences that get closest to 50 tokens
   - Must be complete sentences (no partial sentences in overlap)
```

### Step 4: Progress Tracking
- Track which sentences have been consumed
- Ensure forward progress (at least one new sentence per chunk)
- Handle edge cases: very long sentences, very short text, etc.

## Key Differences from Previous Contract

| Aspect | Old Contract | Option A Contract |
|--------|-------------|-------------------|
| Overlap | "Exactly 50 tokens" | "Closest to 50 tokens" (target) |
| Strictness | Tried to enforce exact 50 with tolerance | Acknowledges variation is inevitable |
| Test Assertion | `abs(overlap - 50) <= 5` | `abs(overlap - 50) <= 25` (wider tolerance) |
| Documentation | "exactly 50-token overlap" | "sentence-aligned overlap closest to 50 tokens" |

## Test Expectations

### VAL-DATA-003: Sentence Boundaries
- **Assertion:** No chunk ends mid-sentence
- **Verification:** All chunks end with `.`, `!`, or `?`

### VAL-DATA-004: Chunk Size
- **Assertion:** Each chunk has at most 500 tokens
- **Verification:** `token_count <= 500`

### VAL-DATA-005: Overlap (Option A)
- **Assertion:** Overlap is sentence-aligned and within reasonable range of 50
- **Verification:** `abs(overlap_tokens - 50) <= 25` (allows 25-75 token range)
- **Note:** This acknowledges that sentence boundaries prevent exact 50-token overlap

## Wikipedia-Specific Improvements

The chunker includes special handling for Wikipedia formatting:

1. **Heading Detection**
   - Lines starting/ending with `=` are section headers
   - Headers are preserved as sentence boundaries
   - Prevents chunks from spanning unrelated sections

2. **Blank Line Segmentation**
   - Double newlines indicate logical breaks
   - Treated as implicit sentence boundaries

3. **Formula Handling**
   - LaTeX blocks (e.g., `$...$`, `$$...$$`) are atomic
   - Never split a formula across chunks

4. **List Item Boundaries**
   - Bullet points (`*`, `-`) and numbered items start new segments
   - List items are treated as individual sentences

## Rationale

### Why Option A?

The previous "exactly 50 tokens" requirement created an impossible constraint:
- Sentence boundaries are unpredictable (sentences vary from 5-100+ tokens)
- You cannot have both "exactly 50 tokens" AND "never cut sentences"

Option A resolves this by:
1. Prioritizing sentence integrity (no mid-sentence cuts)
2. Treating 50 tokens as a target, not a guarantee
3. Accepting that overlap will vary based on sentence lengths

This is the correct trade-off for RAG applications:
- Sentence integrity ensures meaningful chunks
- Approximate overlap preserves context between chunks
- The 50-token target provides reasonable context continuity

## Implementation Notes

- Uses `transformers.AutoTokenizer` for consistent token counting
- Sentence detection uses regex (not NLTK) to minimize dependencies
- Emergency truncation available for pathological cases
- Overlap calculation minimizes `|actual - 50|` while respecting sentence boundaries
