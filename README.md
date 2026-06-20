<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Context Engineering](https://img.shields.io/badge/Context%20Engineering-2026%20Trend-BF91F3?style=flat-square)
![Token Optimization](https://img.shields.io/badge/Token-Optimization-38BDAE?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-70A5FD?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-9%2F9%20Passing-brightgreen?style=flat-square)

**Cut LLM input tokens by 30–60% — without losing the information that matters**

[Why This Matters](#-why-this-matters) · [Demo](#-demo) · [Installation](#️-installation) · [How it Works](#️-how-it-works)

</div>

---

## 🔥 Why This Matters

**Context engineering is the dominant optimization pattern in AI agents right now.** As agent workflows pull in logs, tool outputs, and RAG chunks, raw context bloats fast — burning tokens, money, and the model's attention on noise instead of signal.

This tool sits between your data sources and your LLM call, automatically:
- Removing duplicate lines and repeated log spam
- Folding 500 identical error lines into one line + a count
- Stripping filler phrases that add tokens with zero information value
- Ranking sentences by information density and keeping only the best ones
- Deduplicating overlapping facts across multiple RAG chunks

> **Real impact:** a 500-line server log with repeated errors compressed to **38% of its original token count** in testing — same information, far fewer tokens.

---

## 📸 Demo

**Before (server logs):**
```
2026-06-18T10:01:03 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 1
2026-06-18T10:01:04 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 2
2026-06-18T10:01:05 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 3
2026-06-18T10:01:06 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 4
```

**After:**
```
2026-06-18T10:01:03 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 1   [×4 similar occurrences]
```

```
📊 COMPRESSION REPORT
────────────────────────────────────────
  Detected type        : logs
  Original tokens      : 382
  Compressed tokens     : 147
  Tokens saved          : 235  (61.5% reduction)
  Est. cost saved       : $0.000705
────────────────────────────────────────
```

---

## ⚙️ How it Works

```
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
│  Raw Context  │───▶│  Auto-detect  │───▶│  Apply pipeline: │
│  (logs/docs/  │    │  content type │    │  dedupe → fold   │
│  RAG chunks)  │    │               │    │  → prune → rank  │
└──────────────┘    └───────────────┘    └────────┬─────────┘
                                                   │
                                                   ▼
                                        ┌──────────────────┐
                                        │  Compressed text  │
                                        │  + savings report │
                                        └──────────────────┘
```

| Technique | Best for | What it does |
|-----------|----------|---------------|
| **Deduplication** | All content | Removes exact/near-duplicate lines |
| **Log pattern folding** | Server logs | Collapses repeated errors into `[×N occurrences]` |
| **Filler pruning** | Prose, reports | Strips phrases like *"it's worth noting that"* |
| **Sentence ranking** | Documentation, RAG | Keeps only the highest information-density sentences |
| **Cross-chunk dedup** | RAG retrieval | Removes facts repeated across multiple chunks |

All compression is **rule-based and runs 100% locally** — no API key, no external calls, no added latency.

---

## 🛠️ Installation

```bash
git clone https://github.com/ramyasreenagarajan/context-compressor.git
cd context-compressor
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Web App
```bash
streamlit run app/app.py
```

### CLI — Demo (no file needed)
```bash
python src/cli.py --demo
```

### CLI — Compress a file
```bash
python src/cli.py --file server.log
python src/cli.py --file report.txt --rank-sentences --keep-ratio 0.5
python src/cli.py --file data.txt --json          # machine-readable stats
python src/cli.py --file data.txt --output clean.txt
```

### As a Python library
```python
from src.compressor import ContextCompressor, compress_rag_chunks

# Compress logs or documents
compressor = ContextCompressor(dedupe=True, fold_logs=True)
result = compressor.compress(raw_log_text, content_type="logs")
print(result["compressed_text"])
print(result["stats"]["compression_ratio"])

# Compress a list of RAG-retrieved chunks (dedupes across chunks)
result = compress_rag_chunks([chunk1, chunk2, chunk3])
```

### Drop into an agent pipeline
```python
# Before sending retrieved context to your LLM:
from src.compressor import compress_rag_chunks

retrieved_chunks = vector_store.retrieve(query, top_k=10)
compressed = compress_rag_chunks(retrieved_chunks)

prompt = f"Context:\n{compressed['compressed_text']}\n\nQuestion: {query}"
# → send `prompt` to your LLM, using fewer tokens than the raw chunks
```

---

## 📁 Project Structure

```
context-compressor/
│
├── src/
│   ├── compressor.py     # Core compression engine (5 techniques)
│   └── cli.py            # CLI interface + built-in demos
│
├── app/
│   └── app.py            # Streamlit web app with before/after view
│
├── tests/
│   └── test_compressor.py  # 9 unit tests, all passing
│
├── requirements.txt
└── README.md
```

---

## 🧪 Tests

```bash
python tests/test_compressor.py
# 9/9 tests passed
```

---

## 🗺️ Roadmap

- [ ] Add semantic deduplication using embeddings (catch paraphrased duplicates)
- [ ] Token counting via `tiktoken` for exact (not estimated) counts
- [ ] LangChain / LlamaIndex integration as a drop-in context transformer
- [ ] Support for JSON/structured log formats
- [ ] Configurable compression presets (aggressive / balanced / conservative)

---

## 👩‍💻 Author

**Ramya Sree Nagarajan**
MSc Artificial Intelligence · Royal Holloway, University of London
IEEE Published Researcher · Python · ML · Cybersecurity

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/ramya-sree-nagarajan-619245345)
[![Email](https://img.shields.io/badge/Email-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:ramyasreenagarajan@gmail.com)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
