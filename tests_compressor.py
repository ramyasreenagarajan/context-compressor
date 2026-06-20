"""
Tests for Context Compressor
Run: python -m pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.compressor import (
    ContextCompressor, estimate_tokens, deduplicate_lines,
    fold_repeated_patterns, prune_low_signal_phrases,
    rank_and_filter_sentences, compress_rag_chunks
)


def test_estimate_tokens():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("a" * 400) == 100


def test_deduplicate_lines():
    text = "line one\nline two\nline one\nline three"
    result, removed = deduplicate_lines(text)
    assert removed == 1
    assert result.count("line one") == 1


def test_fold_repeated_patterns():
    text = "\n".join([f"ERROR timeout at attempt {i}" for i in range(10)])
    result, folded = fold_repeated_patterns(text)
    assert folded > 0
    assert "similar occurrences" in result


def test_prune_low_signal_phrases():
    text = "As I mentioned earlier, this is really very important basically."
    result, removed = prune_low_signal_phrases(text)
    assert removed > 0
    assert "mentioned earlier" not in result.lower()


def test_rank_and_filter_sentences():
    text = (
        "The cat sat on the mat. " * 1 +
        "Revenue increased by 50 percent this quarter due to strong sales. " +
        "The weather was nice yesterday. " +
        "Profit margins expanded significantly across all divisions. " +
        "Someone walked the dog in the park."
    )
    result, removed = rank_and_filter_sentences(text, keep_ratio=0.4)
    assert len(result) < len(text)
    assert "Revenue" in result or "Profit" in result


def test_context_compressor_logs():
    compressor = ContextCompressor()
    logs = "\n".join([f"2026-06-18T10:00:0{i%9} ERROR timeout" for i in range(20)])
    result = compressor.compress(logs, content_type="logs")
    assert result["stats"]["compressed_tokens"] < result["stats"]["original_tokens"]
    assert result["stats"]["compression_ratio"] > 0


def test_context_compressor_prose():
    compressor = ContextCompressor(rank_sentences=True, keep_ratio=0.5)
    prose = (
        "Basically, the results were really good. " * 2 +
        "Revenue grew by 30 percent year over year driven by new product lines. " +
        "Essentially, the team worked very hard on this important project. " +
        "Customer satisfaction scores reached an all time high of 95 percent."
    )
    result = compressor.compress(prose, content_type="prose")
    assert result["stats"]["compressed_tokens"] <= result["stats"]["original_tokens"]


def test_compress_rag_chunks():
    chunks = [
        "The capital of France is Paris. Paris has a population of over 2 million.",
        "Paris is the capital of France. It is known for the Eiffel Tower.",
        "France's economy is the seventh largest in the world by GDP.",
    ]
    result = compress_rag_chunks(chunks)
    assert result["stats"]["num_chunks_input"] <= len(chunks)
    assert len(result["compressed_text"]) > 0


def test_no_crash_on_empty_input():
    compressor = ContextCompressor()
    result = compressor.compress("", content_type="auto")
    assert result["compressed_text"] == ""


if __name__ == "__main__":
    import inspect
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and inspect.isfunction(obj)]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} — {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
