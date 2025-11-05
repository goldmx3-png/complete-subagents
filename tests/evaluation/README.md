# RAG Retrieval Evaluation

Simple evaluation scripts to test and compare different retrieval configurations.

## Test Scripts

### 1. Retrieval Accuracy Evaluation
**File**: `evaluate_retrieval.py`

Tests different retrieval configurations and compares accuracy:
- Baseline (Vector-only search)
- Hybrid Search (Vector + BM25)
- Hybrid + Reranking
- Vector + Reranking

**Usage**:
```bash
# Make sure you're in the project root
cd /path/to/complete-subagents

# Run evaluation
python tests/evaluation/evaluate_retrieval.py
```

**Output**:
- Console: Summary statistics and comparison table
- CSV files: Detailed results for each configuration in `tests/evaluation/results/`

**Metrics**:
- Number of results retrieved
- Average relevance score
- Top score (best match)
- Retrieval time

### 2. Chunk Size Comparison
**File**: `compare_chunk_sizes.py`

Tests different token-based chunk sizes (400, 600, 800 tokens).

**Usage**:
```bash
python tests/evaluation/compare_chunk_sizes.py
```

**Output**:
- Console: Chunk statistics for each size
- JSON file: Results in `tests/evaluation/results/chunk_size_comparison.json`

**Metrics**:
- Number of chunks created
- Average tokens per chunk
- Min/max token range

## Test Data

### Test Queries (`test_queries.json`)
Contains 10 representative banking queries covering:
- Simple factual queries
- Procedural questions
- Table lookups
- Complex multi-step procedures

**Categories**:
- `simple_factual`: Direct fact retrieval
- `procedural`: Step-by-step instructions
- `complex_factual`: Multi-condition queries
- `table_lookup`: Information from tables
- `complex_procedural`: Multi-step with context

## Configuration Testing

### Enable/Disable Features

**Hybrid Search**:
```bash
# In .env file
ENABLE_HYBRID_SEARCH=true  # or false
```

**Reranking**:
```bash
ENABLE_RERANKING=true  # or false
```

**Chunk Size**:
```bash
CHUNK_SIZE_TOKENS=600  # 400, 600, or 800
USE_TOKEN_BASED_CHUNKING=true
```

**Semantic Chunking**:
```bash
USE_SEMANTIC_CHUNKING=false  # Enable for LLM-based chunking
```

## Expected Results

Based on 2025 research:

**Hybrid Search**:
- 8-15% accuracy improvement over vector-only
- Slightly slower (50-100ms added latency)

**Reranking**:
- 3-5% additional accuracy boost
- 100-200ms added latency with bge-reranker-large

**Optimal Chunk Size** (for complex banking docs):
- **600 tokens**: Best balance for documents with tables
- 400 tokens: Better precision for simple queries
- 800 tokens: Better context for policy documents

## Interpreting Results

### Score Interpretation
- **0.8+**: Highly relevant
- **0.6-0.8**: Relevant
- **0.4-0.6**: Partially relevant
- **< 0.4**: Low relevance (filtered by MIN_SIMILARITY_SCORE)

### Performance Targets
- **Accuracy**: Avg score > 0.65
- **Speed**: < 500ms total retrieval time
- **Coverage**: At least 3-5 relevant results per query

## Adding Custom Test Queries

Edit `test_queries.json`:
```json
{
  "id": 11,
  "query": "Your custom query",
  "category": "simple_factual",
  "expected_topics": ["keyword1", "keyword2"]
}
```

## Troubleshooting

**No results returned**:
- Check if documents are indexed in Qdrant
- Lower MIN_SIMILARITY_SCORE in .env
- Verify embeddings are generated correctly

**Slow performance**:
- Disable reranking for faster results
- Reduce RERANKER_TOP_K value
- Use CPU instead of CUDA if GPU memory is limited

**Import errors**:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Run from project root directory

## Next Steps

1. Run baseline evaluation to establish metrics
2. Enable hybrid search and re-evaluate
3. Add reranking and compare
4. Test different chunk sizes with your actual documents
5. Choose optimal configuration based on accuracy vs speed trade-off
