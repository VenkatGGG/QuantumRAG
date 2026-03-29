---
name: test-worker
description: Worker for mathematical verification and test suite implementation
---

# Test Worker

Handles mathematical verification tests for core algorithms.

## When to Use This Skill

Use for features involving:
- Mathematical verification of algorithms
- Deterministic output testing
- Baseline vector testing
- pytest suite creation

## Required Skills

None

## Work Procedure

1. **Create test suite**
   - Create test_suite.py with pytest
   - Test mean pooling with mocked tensors
   - Test cosine similarity with known vectors
   - Verify deterministic outputs

2. **Implement mathematical tests**
   - Use baseline vectors with known expected outputs
   - Test edge cases: zero vectors, identical vectors, orthogonal vectors
   - Verify floating-point precision

3. **Run and verify**
   - Run full test suite: `python -m pytest test_suite.py -v`
   - All tests must pass
   - Document expected values

4. **Commit work**

## Example Handoff

```json
{
  "salientSummary": "Created comprehensive test_suite.py with mathematical verification of mean pooling and cosine similarity functions using baseline vectors with deterministic expected outputs.",
  "whatWasImplemented": "Created test_suite.py with pytest. Implemented test_mean_pooling_baseline() verifying E = sum(T_i * M_i) / max(sum(M_i), epsilon) with mocked 2x3x384 tensor. Implemented test_cosine_similarity_baseline() verifying similarity = dot(A,B)/(norm(A)*norm(B)) with known vectors. Added tests for edge cases: identical vectors (similarity=1), orthogonal vectors (similarity=0), zero vector handling. All tests use deterministic mocked data to ensure reproducible results.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "python -m pytest test_suite.py -v", "exitCode": 0, "observation": "12 tests passed: 3 mean pooling tests, 5 cosine similarity tests, 4 edge case tests"},
      {"command": "python -m pytest test_suite.py::test_mean_pooling_baseline -v", "exitCode": 0, "observation": "Mean pooling produces expected tensor [0.5, 0.5, 0.5, ...] for baseline input"},
      {"command": "python -m pytest test_suite.py::test_cosine_similarity_baseline -v", "exitCode": 0, "observation": "Cosine similarity returns 0.9999999 for nearly identical vectors, 0.0 for orthogonal vectors"}
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {"file": "test_suite.py", "cases": [
        {"name": "test_mean_pooling_baseline", "verifies": "Mean pooling formula produces expected output for baseline tensor"},
        {"name": "test_mean_pooling_attention_mask", "verifies": "Attention mask correctly zeros out padding tokens"},
        {"name": "test_mean_pooling_epsilon", "verifies": "Epsilon prevents division by zero"},
        {"name": "test_cosine_similarity_baseline", "verifies": "Cosine similarity formula produces expected output"},
        {"name": "test_cosine_similarity_identical", "verifies": "Identical vectors have similarity 1.0"},
        {"name": "test_cosine_similarity_orthogonal", "verifies": "Orthogonal vectors have similarity 0.0"},
        {"name": "test_cosine_similarity_opposite", "verifies": "Opposite vectors have similarity -1.0"},
        {"name": "test_cosine_similarity_normalized", "verifies": "Normalized vectors preserve similarity ratios"},
        {"name": "test_deterministic_mean_pooling", "verifies": "Same input always produces same output"},
        {"name": "test_deterministic_cosine_similarity", "verifies": "Same vectors always produce same similarity"},
        {"name": "test_floating_point_precision", "verifies": "Results match expected within 1e-6 tolerance"},
        {"name": "test_edge_case_zero_vector", "verifies": "Zero vector handled gracefully"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Mathematical formulas produce unexpected results
- Floating-point precision issues
- Mock tensor creation fails
