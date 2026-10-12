## Retrieval (150 answerable questions, section-level)

| Run | Retrieval | Embedding | Chunk | recall@5 | hit@5 (pooled) | MRR |
|---|---|---|---|---|---|---|
| vector-c300 | vector | bge-small-en-v1.5 | 300 | 0.793 | 0.940 | 0.806 |
| vector-c800 | vector | bge-small-en-v1.5 | 800 | 0.803 | 0.933 | 0.796 |
| bm25-c300 | bm25 | bge-small-en-v1.5 | 300 | 0.787 | 0.893 | 0.742 |
| bm25-c500 | bm25 | bge-small-en-v1.5 | 500 | 0.807 | 0.900 | 0.742 |
| bm25-c800 | bm25 | bge-small-en-v1.5 | 800 | 0.813 | 0.907 | 0.752 |
| hybrid-c300 | hybrid | bge-small-en-v1.5 | 300 | 0.860 | 0.953 | 0.835 |
| hybrid-c500 | hybrid | bge-small-en-v1.5 | 500 | 0.880 | 0.967 | 0.848 |
| hybrid-c800 | hybrid | bge-small-en-v1.5 | 800 | 0.877 | 0.960 | 0.841 |
| rerank-c300 | rerank | bge-small-en-v1.5 | 300 | 0.853 | 0.973 | 0.885 |
| rerank-c500 | rerank | bge-small-en-v1.5 | 500 | 0.860 | 0.973 | 0.885 |
| rerank-c800 | rerank | bge-small-en-v1.5 | 800 | 0.860 | 0.973 | 0.876 |
| rerank-bge-reranker-base-c500 | rerank | bge-small-en-v1.5 | 500 | 0.873 | 0.953 | 0.871 |
| finetuned | vector | bge-small-dhan | 500 | 0.850 | 0.967 | 0.832 |
| finetuned-hybrid | hybrid | bge-small-dhan | 500 | 0.887 | 0.980 | 0.845 |
| vector-c500 | vector | bge-small-en-v1.5 | 500 | 0.830 | 0.953 | 0.809 |

## End to end (150 answerable + 20 unanswerable)

| Run | Retrieval | Answer model | Grader | Correct | Faithful | Refusal acc | False refusal | LLM calls / q | p50 s | p95 s | $ / 1K q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | vector | llama-3.3-70b-instruct | anthropic/claude-sonnet-5.5 | 0.847 | 0.931 | 1.000 | 0.033 | – | 0.54 | 2.19 | 0.30 |
| longctx-llama3.3-70b | none (full doc) | llama-3.3-70b-instruct | claude-opus-5.5 | 0.540 | 0.187 | 0.000 | 0.000 | 1.0 | 4.61 | 45.76 | 23.85 |
| hybrid | hybrid | llama-3.3-70b-instruct | claude-opus-5.5 | 0.913 | 0.980 | 0.900 | 0.013 | 1.0 | 3.10 | 12.78 | 0.19 |
| agent | hybrid | llama-3.3-70b-instruct | claude-opus-5.5 | 0.773 | 0.992 | 1.000 | 0.193 | 3.6 | 8.22 | 28.60 | 0.59 |
| vector-rag | vector | llama-3.3-70b-instruct | claude-opus-5.5 | 0.860 | 0.965 | 1.000 | 0.040 | 1.0 | 2.45 | 6.77 | 0.23 |
| agent-v2 | hybrid | llama-3.3-70b-instruct | claude-opus-5.5 | 0.887 | 0.986 | 1.000 | 0.027 | 3.4 | 7.37 | 23.00 | 0.59 |
