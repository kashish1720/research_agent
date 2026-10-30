# Manual answer review sheet

Pipeline support markers are self-reported, not independent human judgments.

## Offline summary

| Sample | Findings | Citation integrity | Support markers | Distinct domains | Snippet-only share |
| --- | --- | --- | --- | --- | --- |
| memory_approaches | 7 | not available: no saved evidence ledger or resolvable source citations | supported=7, partial=0, unchecked=0 (pipeline self-check, not independent) | 4 | 2/4 (50%) |
| multi_agent | 8 | not available: no saved evidence ledger or resolvable source citations | supported=8, partial=0, unchecked=0 (pipeline self-check, not independent) | 5 | 0/5 (0%) |
| search_faults | 3 | not available: no saved evidence ledger or resolvable source citations | supported=3, partial=0, unchecked=0 (pipeline self-check, not independent) | 2 | 2/2 (100%) |
| vector_db | 7 | not available: no saved evidence ledger or resolvable source citations | supported=7, partial=0, unchecked=0 (pipeline self-check, not independent) | 5 | 3/5 (60%) |
| vector_db_degraded | 6 | not available: no saved evidence ledger or resolvable source citations | supported=6, partial=0, unchecked=0 (pipeline self-check, not independent) | 3 | 2/3 (67%) |

## Candidate-name review aids

- **memory_approaches:** Short, Long, Hybrid, Approximate, NN, Vector, Scalability, Integration
- **multi_agent:** Distributed, AI, Parallelism, An, Supervisor, The, High, Single, Tool
- **search_faults:** Qdrant, RAG, Weaviate, Milvus
- **vector_db:** Qdrant, RAG, Weaviate, AI, Milvus, ElasticSearch, OpenSearch, Pinecone, Chroma, Rust, Apache, API
- **vector_db_degraded:** Qdrant, Rust, Milvus, GPU, RAG, SSD, Performance, Pinecone, Major, LangChain, LlamaIndex

## Finding-level human review

| Sample | Finding # | Finding text | Cited evidence IDs | Pipeline support | Cited source URLs | Human verdict (Supported / Partly / Not supported / Cannot check) | Note |
|---|---:|---|---|---|---|---|---|
| memory_approaches | 1 | Short‑term memory provides immediate task context within the active window. | E1 | supported | not available: no saved evidence-to-source mapping |  |  |
| memory_approaches | 2 | Long‑term persistent memory enables agents to store, retrieve, and learn from past interactions across sessions. | E2, E3 | supported | not available: no saved evidence-to-source mapping |  |  |
| memory_approaches | 3 | Hybrid vector‑plus‑graph memory systems deliver a balance of speed and depth for retrieval. | E5 | supported | not available: no saved evidence-to-source mapping |  |  |
| memory_approaches | 4 | Approximate k‑NN retrieval trades accuracy for speed, returning results in milliseconds. | E6 | supported | not available: no saved evidence-to-source mapping |  |  |
| memory_approaches | 5 | Vector databases have tradeoffs between accuracy, speed, and memory usage. | E7 | supported | not available: no saved evidence-to-source mapping |  |  |
| memory_approaches | 6 | Scalability requires pruning and tiered storage to control costs as memory volumes grow. | E8 | supported | not available: no saved evidence-to-source mapping |  |  |
| memory_approaches | 7 | Integration complexity arises from coordinating short‑term, long‑term, vector, and graph memory layers. | E9 | supported | not available: no saved evidence-to-source mapping |  |  |
| multi_agent | 1 | Distributed AI agents separate tasks among specialized agents working in parallel. | E1 | supported | not available: no saved evidence-to-source mapping |  |  |
| multi_agent | 2 | Parallelism at scale reduces latency and increases throughput for high‑volume workflows. | E2 | supported | not available: no saved evidence-to-source mapping |  |  |
| multi_agent | 3 | An AI Supervisor coordinates agents, enforces policies, and escalates to humans. | E3 | supported | not available: no saved evidence-to-source mapping |  |  |
| multi_agent | 4 | The initial setup of multi‑agent systems is complex and time‑consuming. | E5 | supported | not available: no saved evidence-to-source mapping |  |  |
| multi_agent | 5 | High‑quality, up‑to‑date data is essential to avoid system errors. | E6 | supported | not available: no saved evidence-to-source mapping |  |  |
| multi_agent | 6 | Single‑agent systems are centralized, while multi‑agent systems are distributed or coordinated. | E7 | supported | not available: no saved evidence-to-source mapping |  |  |
| multi_agent | 7 | Single agents manage all steps, whereas multi‑agents cooperate via messages or a shared orchestrator. | E8 | supported | not available: no saved evidence-to-source mapping |  |  |
| multi_agent | 8 | Tool‑heavy tasks can suffer a 2x to 6x efficiency penalty in multi‑agent systems due to coordination overhead and context fragmentation. | E9 | supported | not available: no saved evidence-to-source mapping |  |  |
| search_faults | 1 | Qdrant is highlighted as a suitable open‑source vector database for production RAG applications. | E1 | supported | not available: no saved evidence-to-source mapping |  |  |
| search_faults | 2 | Weaviate is identified as an open‑source vector database suitable for RAG. | E2 | supported | not available: no saved evidence-to-source mapping |  |  |
| search_faults | 3 | Milvus is described as an open‑source vector database for large‑scale RAG deployments. | E3 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db | 1 | Qdrant is highlighted as a strong fit for production RAG applications. | E1 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db | 2 | Weaviate is described as an AI‑native database with built‑in vectorization and multimodal support. | E2 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db | 3 | Milvus supports large‑scale embedding similarity search. | E5 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db | 4 | ElasticSearch and OpenSearch underperform for high‑dimensional vector similarity searches compared to dedicated vector databases. | E4 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db | 5 | Pinecone’s free plan limits scalability, but it is not open‑source and thus not considered in the top three. | E3 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db | 6 | Chroma’s core is written in Rust, licensed Apache 2.0, and starts in memory with four API calls. | E7 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db | 7 | OpenSearch, licensed Apache 2.0, stores vectors in knn_vector fields. | E8 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db_degraded | 1 | Qdrant is an open‑source Rust vector database known for real‑time embedding search and advanced filtering. | E1 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db_degraded | 2 | Milvus is an open‑source, GPU‑accelerated vector database optimized for large‑scale, high‑throughput use. | E2 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db_degraded | 3 | Milvus is identified as the leading open‑source vector database for RAG workloads. | E5 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db_degraded | 4 | Milvus now supports SSD‑optimized vector indexing, reducing memory usage while preserving performance. | E6 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db_degraded | 5 | Performance testing shows 6‑9× better performance than Pinecone in RAG workloads. | E3 | supported | not available: no saved evidence-to-source mapping |  |  |
| vector_db_degraded | 6 | Major RAG frameworks such as LangChain and LlamaIndex have first‑class Pinecone support, indicating a mature integration ecosystem. | E4 | supported | not available: no saved evidence-to-source mapping |  |  |
