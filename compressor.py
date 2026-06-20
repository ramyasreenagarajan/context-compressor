"""
Context Compressor — Core Engine
===================================
Author : Ramya Sree Nagarajan
GitHub : github.com/ramyasreenagarajan

Compresses logs, documents, and RAG chunks before they reach an LLM —
cutting token usage and cost while preserving the information that matters.

Techniques used:
  1. Deduplication       — remove repeated lines/blocks
  2. Stopword pruning     — strip low-signal filler words
  3. Sentence ranking     — keep only the most information-dense sentences
  4. Log pattern folding  — collapse repeated log patterns into counts
  5. Whitespace/structure normalization
"""

import re
import hashlib
from collections import Counter, OrderedDict
from typing import List, Dict, Tuple
import math


# ── Token Estimation ──────────────────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (no API needed).
    Rule of thumb: ~4 characters per token for English text.
    """
    return max(1, len(text) // 4)


# ── Technique 1: Deduplication ────────────────────────────────────────────────
def deduplicate_lines(text: str, similarity_threshold: float = 0.92) -> Tuple[str, int]:
    """Remove exact and near-duplicate lines. Returns (text, lines_removed)."""
    lines = text.split("\n")
    seen_hashes = set()
    kept = []
    removed = 0

    for line in lines:
        normalized = re.sub(r"\s+", " ", line.strip().lower())
        if not normalized:
            kept.append(line)
            continue
        h = hashlib.md5(normalized.encode()).hexdigest()
        if h in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(h)
        kept.append(line)

    return "\n".join(kept), removed


# ── Technique 2: Log Pattern Folding ──────────────────────────────────────────
def fold_repeated_patterns(text: str) -> Tuple[str, int]:
    """
    Detect repeated log-line patterns (e.g. same error logged 500 times)
    and collapse them into a single line with a count.
    Example:
      ERROR: timeout at 10:01   x412
    """
    lines = text.split("\n")

    def normalize_for_pattern(line: str) -> str:
        # Replace numbers, timestamps, IDs with placeholders to detect patterns
        line = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "<TIMESTAMP>", line)
        line = re.sub(r"\b\d+\.\d+\.\d+\.\d+\b", "<IP>", line)
        line = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                       "<UUID>", line)
        line = re.sub(r"\b\d+\b", "<NUM>", line)
        return line.strip()

    pattern_counts = Counter()
    pattern_to_original = {}
    for line in lines:
        if not line.strip():
            continue
        pattern = normalize_for_pattern(line)
        pattern_counts[pattern] += 1
        if pattern not in pattern_to_original:
            pattern_to_original[pattern] = line

    folded = []
    folded_count = 0
    emitted_patterns = set()
    for line in lines:
        if not line.strip():
            folded.append(line)
            continue
        pattern = normalize_for_pattern(line)
        count = pattern_counts[pattern]
        if count >= 3:
            if pattern not in emitted_patterns:
                folded.append(f"{pattern_to_original[pattern]}   [×{count} similar occurrences]")
                emitted_patterns.add(pattern)
                folded_count += (count - 1)
            # else: skip — already folded
        else:
            folded.append(line)

    return "\n".join(folded), folded_count


# ── Technique 3: Stopword Pruning ─────────────────────────────────────────────
LOW_SIGNAL_PHRASES = [
    r"\bas (?:i|we) (?:mentioned|said|noted)( earlier| before| above)?\b",
    r"\bit(?:'s| is) (?:worth noting|important to note) that\b",
    r"\bin other words\b",
    r"\bto put it (?:simply|another way)\b",
    r"\bneedless to say\b",
    r"\bat the end of the day\b",
    r"\bfor what it'?s worth\b",
    r"\bjust to (?:clarify|be clear)\b",
    r"\bI think (?:that )?\b",
    r"\bbasically\b",
    r"\bessentially\b",
    r"\bin fact\b",
    r"\breally\b(?=\s)",
    r"\bvery\b(?=\s)",
    r"\bquite\b(?=\s)",
]

def prune_low_signal_phrases(text: str) -> Tuple[str, int]:
    """Remove filler phrases that add tokens without adding information."""
    original_len = len(text)
    for pattern in LOW_SIGNAL_PHRASES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    # Clean up leading punctuation/whitespace left after a phrase was stripped
    text = re.sub(r"(?m)^[,\s]+", "", text)
    text = re.sub(r"([.!?])\s*,\s*", r"\1 ", text)
    chars_removed = original_len - len(text)
    return text.strip(), chars_removed


# ── Technique 4: Sentence Ranking (Extractive Summarization) ────────────────
def rank_and_filter_sentences(text: str, keep_ratio: float = 0.6) -> Tuple[str, int]:
    """
    Score sentences by word-frequency importance (TF-based) and keep only
    the top `keep_ratio` fraction. Good for long prose/documentation/RAG chunks.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(sentences) <= 3:
        return text, 0  # too short to bother

    # Word frequency scoring
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    freq = Counter(words)
    max_freq = max(freq.values()) if freq else 1
    for w in freq:
        freq[w] = freq[w] / max_freq

    STOPWORDS = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "has", "had", "was", "were", "this", "that", "with", "from",
        "they", "their", "have", "been", "would", "could", "should",
    }

    scored = []
    for i, sent in enumerate(sentences):
        sent_words = re.findall(r"\b[a-z]{3,}\b", sent.lower())
        sent_words = [w for w in sent_words if w not in STOPWORDS]
        score = sum(freq.get(w, 0) for w in sent_words)
        score = score / max(len(sent_words), 1)
        # Boost sentences with numbers/data (often important facts)
        if re.search(r"\d", sent):
            score *= 1.3
        # Slight boost for position (first/last sentences often key)
        if i == 0 or i == len(sentences) - 1:
            score *= 1.15
        scored.append((score, i, sent))

    n_keep = max(3, int(len(sentences) * keep_ratio))
    top = sorted(scored, key=lambda x: -x[0])[:n_keep]
    top_sorted_by_position = sorted(top, key=lambda x: x[1])  # restore original order

    result = " ".join(s for _, _, s in top_sorted_by_position)
    removed = len(sentences) - len(top_sorted_by_position)
    return result, removed


# ── Technique 5: Whitespace Normalization ─────────────────────────────────────
def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


# ── Main Compressor Class ─────────────────────────────────────────────────────
class ContextCompressor:
    """
    Orchestrates all compression techniques in a configurable pipeline.
    """

    def __init__(self,
                 dedupe: bool = True,
                 fold_logs: bool = True,
                 prune_filler: bool = True,
                 rank_sentences: bool = False,
                 keep_ratio: float = 0.6):
        self.dedupe         = dedupe
        self.fold_logs      = fold_logs
        self.prune_filler   = prune_filler
        self.rank_sentences = rank_sentences
        self.keep_ratio     = keep_ratio

    def compress(self, text: str, content_type: str = "auto") -> Dict:
        """
        Compress text and return a report with stats + compressed output.

        content_type: "auto" | "logs" | "prose" | "rag_chunks"
        """
        original_tokens = estimate_tokens(text)
        original_chars  = len(text)
        stats = {"steps": []}

        result = text

        # Auto-detect content type
        if content_type == "auto":
            content_type = self._detect_content_type(result)
        stats["detected_type"] = content_type

        # Step 1: Deduplication
        if self.dedupe:
            result, removed = deduplicate_lines(result)
            stats["steps"].append({"step": "deduplication", "lines_removed": removed})

        # Step 2: Log folding (only for log-like content)
        if self.fold_logs and content_type == "logs":
            result, folded = fold_repeated_patterns(result)
            stats["steps"].append({"step": "log_pattern_folding", "occurrences_folded": folded})

        # Step 3: Filler pruning
        if self.prune_filler:
            result, chars_removed = prune_low_signal_phrases(result)
            stats["steps"].append({"step": "filler_pruning", "chars_removed": chars_removed})

        # Step 4: Sentence ranking (only for prose/RAG content)
        if self.rank_sentences and content_type in ("prose", "rag_chunks"):
            result, sents_removed = rank_and_filter_sentences(result, self.keep_ratio)
            stats["steps"].append({"step": "sentence_ranking", "sentences_removed": sents_removed})

        # Step 5: Whitespace cleanup (always last)
        result = normalize_whitespace(result)

        compressed_tokens = estimate_tokens(result)
        compressed_chars  = len(result)

        stats.update({
            "original_tokens":    original_tokens,
            "compressed_tokens":  compressed_tokens,
            "tokens_saved":       original_tokens - compressed_tokens,
            "compression_ratio":  round(1 - (compressed_tokens / max(original_tokens, 1)), 4),
            "original_chars":     original_chars,
            "compressed_chars":   compressed_chars,
            "estimated_cost_saved_usd": round(
                (original_tokens - compressed_tokens) * 0.000003, 6
            ),  # rough $3/1M input token estimate
        })

        return {
            "compressed_text": result,
            "stats": stats,
        }

    @staticmethod
    def _detect_content_type(text: str) -> str:
        """Heuristically classify input as logs, prose, or rag_chunks."""
        log_indicators = len(re.findall(
            r"(ERROR|WARN|INFO|DEBUG|\d{4}-\d{2}-\d{2}|Exception|Traceback)",
            text))
        chunk_markers = len(re.findall(r"\[Chunk \d+\]|---", text))

        lines = text.split("\n")
        avg_line_len = sum(len(l) for l in lines) / max(len(lines), 1)

        if log_indicators > 5 and avg_line_len < 150:
            return "logs"
        elif chunk_markers > 2:
            return "rag_chunks"
        else:
            return "prose"


# ── Batch compression for multiple RAG chunks ────────────────────────────────
def compress_rag_chunks(chunks: List[str],
                          max_chunks: int = None,
                          dedupe_across_chunks: bool = True) -> Dict:
    """
    Compress a list of RAG-retrieved chunks together — removing redundancy
    ACROSS chunks (common when multiple chunks repeat the same fact).
    """
    compressor = ContextCompressor(rank_sentences=True, keep_ratio=0.7)

    if dedupe_across_chunks:
        seen_sentences = set()
        deduped_chunks = []
        for chunk in chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            unique_sentences = []
            for sent in sentences:
                key = re.sub(r"\s+", " ", sent.strip().lower())
                if key and key not in seen_sentences:
                    seen_sentences.add(key)
                    unique_sentences.append(sent)
            if unique_sentences:
                deduped_chunks.append(" ".join(unique_sentences))
        chunks = deduped_chunks

    if max_chunks:
        chunks = chunks[:max_chunks]

    combined = "\n---\n".join(chunks)
    result = compressor.compress(combined, content_type="rag_chunks")
    result["stats"]["num_chunks_input"]  = len(chunks)
    result["stats"]["num_chunks_output"] = len(result["compressed_text"].split("\n---\n"))
    return result
