The pipeline's own support markers are a self-check, not an independent verification.

# Evaluation details

The observations below use `evals/results.json` for expected statuses and measured metrics, and the corresponding `samples/*.json` files for answer excerpts and source/domain context. `PASS` means only that the final status belongs to the case's expected-status list; it is not an answer-quality or unsupported-claim measurement. Timings shown are the evaluation harness's uncached `wall_s`.

## `vector_db`

**Query:** Compare the top 3 open-source vector databases for a startup building RAG products.

- **Expected behavior:** Final status `DONE` or `DEGRADED`.
- **Actual:** `DONE`; confidence `MEDIUM`; 7 findings; 5 sources across 5 domains; coverage `3/3`; 0 retries; 26.8 seconds.
- **Short-answer excerpt:** “The top three open-source vector databases ... are Qdrant, Weaviate, and Milvus. Qdrant is praised for production-ready RAG...”
- **What worked (draft: verify):** The sample returns findings about all three named databases and reports complete sub-question coverage.
- **What did not (draft: verify):** The sample says it has no direct comparative performance benchmarks and reports that 60% of evidence items came from snippets. A DONE status therefore should not be read as confirmation of comparative correctness.

## `multi_agent`

**Query:** Research the pros and cons of using a multi-agent architecture for customer support automation.

- **Expected behavior:** Final status `DONE` or `DEGRADED`.
- **Actual:** `DONE`; confidence `MEDIUM`; 8 findings; 5 sources across 5 domains; coverage `3/3`; 0 retries; 29.7 seconds.
- **Short-answer excerpt:** “A multi-agent architecture can boost throughput and reduce latency by parallelizing tasks... However, it introduces significant complexity...”
- **What worked (draft: verify):** The sample has findings covering advantages, disadvantages, and trade-offs, with all three sub-questions covered.
- **What did not (draft: verify):** The sample limitations say there is no evidence on real-deployment user satisfaction, error rates, or cost and maintenance differences; the disadvantage sub-question lists one independent domain.

## `memory_approaches`

**Query:** Compare different approaches to adding memory in an AI support agent.

- **Expected behavior:** Final status `DONE` or `DEGRADED`.
- **Actual:** `DONE`; confidence `MEDIUM`; 7 findings; 4 sources across 4 domains; coverage `3/3`; 0 retries; 26.1 seconds.
- **Short-answer excerpt:** “AI support agents can add memory through short-term context windows, long-term persistent storage, and hybrid vector-plus-graph systems.”
- **What worked (draft: verify):** The sample covers all sub-questions and includes findings on short-term, persistent, and hybrid memory.
- **What did not (draft: verify):** The sample explicitly notes no direct evidence comparing latency or accuracy across approaches and that 50% of evidence items came from snippets.

## `nonsense`

**Query:** `asdkjh qwpoeiru zxcmnb lkjhg`

- **Expected behavior:** `FAILED_GRACEFULLY`.
- **Actual:** `FAILED_GRACEFULLY`; confidence `LOW`; 0 findings; 0 sources across 0 domains; coverage `0/0`; 0 retries; 2.4 seconds.
- **Short-answer excerpt:** “The request does not contain a clear question or topic to research.”
- **What worked (draft: verify):** The sample declines to invent a research answer, returns no findings or sources, and the evaluation records no search or fetch calls.
- **What did not (draft: verify):** No failure is apparent in the recorded expected behavior or sample; this case does not test research quality for an answerable question.

## `search_faults`

**Query:** Compare the top 3 open-source vector databases for a startup building RAG products. Fault: `timeout:0.6`.

- **Expected behavior:** `DONE`, `DEGRADED`, or `FAILED_GRACEFULLY`.
- **Actual:** `DEGRADED`; confidence `LOW`; 3 findings; 2 sources across 2 domains; coverage `2/3`; 6 retries; 42.2 seconds.
- **Short-answer excerpt:** “The top three open-source vector databases ... are Qdrant, Weaviate, and Milvus... While performance, scalability, and integration details are limited in the evidence...”
- **What worked (draft: verify):** The sample returns some cited findings despite injected search timeouts and records degraded status rather than silently presenting the run as fully complete.
- **What did not (draft: verify):** One sub-question is uncovered; the recorded reasons include budget exhaustion, snippet-only evidence, and uncovered sub-questions. The sample reports 100% snippet-only evidence.

## Prompt-injection tests (offline)

This is covered by offline pytest fixtures, not a row in `evals/results.json`; evaluation metrics are therefore **not applicable**. Run:

```powershell
pytest -q -k "canary or injection" -v
```

Relevant tests:

- `tests/test_pipeline.py::test_injection_canary_is_removed_before_extraction_and_final_json`
- `tests/test_select_fetch_node.py::test_injection_in_fetched_page_is_removed_and_flagged`
- `tests/test_select_fetch_node.py::test_heavy_injection_drops_source_and_traces_note`
- `tests/test_extract_node.py::test_injection_removed_upstream_cannot_be_restored_by_llm`
- `tests/test_filters.py::test_injection_lines_are_removed_and_benign_text_is_unchanged`

## Release bar: checks enforced in code

- Extraction accepts a passage only when its normalized text occurs contiguously in the associated full source text; it also checks configured word-count bounds, duplicates, and per-source evidence caps.
- Synthesis validates that findings cite known evidence IDs and that conflicts cite at least two valid IDs. Final verification drops findings with absent/unknown evidence IDs.
- With strict numeric checking enabled, numeric values and percentages in a finding must occur in that finding's cited passages after normalization. An entailment LLM can also mark findings unsupported (drop) or partial.
- Entailment is not guaranteed to run: when unavailable, findings can remain unchecked and the answer is limited accordingly. The short answer check detects unsupported numbers/percentages/years against surviving finding statements; it does not establish semantic support for ordinary prose.
- These controls establish passage presence and citation-ID integrity, not that a source is accurate, that a quotation is representative, or that every generated claim is true. The evaluation data does not measure unsupported-claim rate.

## Manual review of a sample (human check, not automated)

**Status: not performed (time limit).** A review sheet generator exists (`scripts/review_sheet.py`), but no human verdicts were recorded for this submission. The unsupported-claim rate was therefore not measured, and the pipeline's own support markers (all 'supported' in the samples) are a self-check, not independent verification. Candidate names were not checked manually; the vector database samples mention Pinecone, which is a managed service and not open source.

| Human verdict | Count |
|---|---:|
| Supported | Not done (time limit) |
| Partly supported | Not done (time limit) |
| Not supported | Not done (time limit) |
| Could not check | Not done (time limit) |

unsupported-claim rate = Not supported / (Supported + Partly + Not supported); Could not check is reported separately
