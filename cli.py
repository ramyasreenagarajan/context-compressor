"""
Context Compressor — CLI
===========================
Author : Ramya Sree Nagarajan

Usage:
  python src/cli.py --file logs.txt
  python src/cli.py --file document.txt --rank-sentences --keep-ratio 0.5
  echo "some text" | python src/cli.py --stdin
  python src/cli.py --demo
"""

import argparse
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.compressor import ContextCompressor, estimate_tokens


BANNER = """
╔════════════════════════════════════════════════════════╗
║         🗜️  Context Compressor for LLM Agents         ║
║   Shrink logs, docs & RAG chunks before they hit the   ║
║   model — save tokens, cost, and context window space  ║
║   Built by Ramya Sree Nagarajan · MSc AI · RHUL        ║
╚════════════════════════════════════════════════════════╝
"""

DEMO_LOG = """2026-06-18T10:01:02 INFO  Request received from 192.168.1.45 id=a1b2c3d4-1111-2222-3333-444455556666
2026-06-18T10:01:02 INFO  Request received from 192.168.1.45 id=b2c3d4e5-1111-2222-3333-444455556667
2026-06-18T10:01:03 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 1
2026-06-18T10:01:04 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 2
2026-06-18T10:01:05 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 3
2026-06-18T10:01:06 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 4
2026-06-18T10:01:07 ERROR Connection timeout to upstream service at 10.0.0.5 retrying attempt 5
2026-06-18T10:01:08 WARN  Circuit breaker opened for service payments-api
2026-06-18T10:01:09 INFO  Request received from 192.168.1.50 id=c3d4e5f6-1111-2222-3333-444455556668
2026-06-18T10:01:09 INFO  Request received from 192.168.1.50 id=c3d4e5f6-1111-2222-3333-444455556668
2026-06-18T10:01:10 INFO  Health check passed for service auth-api
2026-06-18T10:01:11 INFO  Health check passed for service auth-api
2026-06-18T10:01:12 INFO  Health check passed for service auth-api
2026-06-18T10:01:13 INFO  Health check passed for service auth-api
2026-06-18T10:01:14 DEBUG Cache hit ratio: 0.87 for key namespace=user_sessions
2026-06-18T10:01:15 ERROR NullPointerException at OrderProcessor.java:142
2026-06-18T10:01:16 ERROR NullPointerException at OrderProcessor.java:142
2026-06-18T10:01:17 ERROR NullPointerException at OrderProcessor.java:142
"""

DEMO_PROSE = """As I mentioned earlier, the quarterly report shows really strong growth
across all business segments. It's worth noting that revenue increased by 23 percent
year over year, driven primarily by the enterprise software division. In other words,
the company is performing very well in a difficult macroeconomic environment.
Basically, the management team attributes this growth to three key factors: improved
sales efficiency, expansion into new geographic markets, and successful product launches.
At the end of the day, customer retention rates also improved significantly, reaching
94 percent for the fiscal year, up from 89 percent in the prior year. Essentially, this
demonstrates strong product-market fit. For what it's worth, the company also reduced
operating expenses by 8 percent through automation initiatives. Needless to say, investors
reacted positively to these results, with the stock price rising 12 percent following
the earnings announcement. To put it simply, this was one of the strongest quarters in
the company's history, and analysts have raised their price targets accordingly."""


def print_report(result: dict):
    stats = result["stats"]
    print("\n" + "─" * 60)
    print(f"  📊 COMPRESSION REPORT")
    print("─" * 60)
    print(f"  Detected content type : {stats['detected_type']}")
    print(f"  Original tokens       : {stats['original_tokens']:,}")
    print(f"  Compressed tokens     : {stats['compressed_tokens']:,}")
    print(f"  Tokens saved           : {stats['tokens_saved']:,}  "
          f"({stats['compression_ratio']*100:.1f}% reduction)")
    print(f"  Est. cost saved        : ${stats['estimated_cost_saved_usd']:.6f}  "
          f"(at $3/1M input tokens)")
    print("─" * 60)
    print(f"  Pipeline steps applied:")
    for step in stats["steps"]:
        name = step["step"].replace("_", " ").title()
        detail = {k: v for k, v in step.items() if k != "step"}
        print(f"    • {name:<22} {detail}")
    print("─" * 60 + "\n")


def run_demo():
    print(BANNER)
    print("🎬  DEMO 1 — Log Compression (server logs with repeated errors)\n")
    compressor = ContextCompressor(dedupe=True, fold_logs=True, prune_filler=False)
    result = compressor.compress(DEMO_LOG, content_type="logs")
    print("BEFORE:")
    print(DEMO_LOG[:300] + "...\n")
    print("AFTER:")
    print(result["compressed_text"])
    print_report(result)

    print("\n🎬  DEMO 2 — Prose Compression (filler-heavy business report)\n")
    compressor2 = ContextCompressor(dedupe=False, fold_logs=False,
                                     prune_filler=True, rank_sentences=True,
                                     keep_ratio=0.6)
    result2 = compressor2.compress(DEMO_PROSE, content_type="prose")
    print("BEFORE:")
    print(DEMO_PROSE[:300] + "...\n")
    print("AFTER:")
    print(result2["compressed_text"])
    print_report(result2)

    print("✅  Demo complete! Try with your own file:")
    print("   python src/cli.py --file your_logs.txt\n")


def main():
    parser = argparse.ArgumentParser(description="Context Compressor for LLM Agents")
    parser.add_argument("--file", type=str, help="Path to text/log file to compress")
    parser.add_argument("--stdin", action="store_true", help="Read input from stdin")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo")
    parser.add_argument("--type", type=str, default="auto",
                        choices=["auto", "logs", "prose", "rag_chunks"],
                        help="Content type hint (default: auto-detect)")
    parser.add_argument("--rank-sentences", action="store_true",
                        help="Enable extractive sentence ranking (for prose)")
    parser.add_argument("--keep-ratio", type=float, default=0.6,
                        help="Fraction of sentences to keep when ranking (default: 0.6)")
    parser.add_argument("--no-dedupe", action="store_true", help="Disable line deduplication")
    parser.add_argument("--no-fold", action="store_true", help="Disable log pattern folding")
    parser.add_argument("--no-prune", action="store_true", help="Disable filler phrase pruning")
    parser.add_argument("--output", type=str, help="Save compressed output to file")
    parser.add_argument("--json", action="store_true", help="Print stats as JSON")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if args.file:
        with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif args.stdin:
        text = sys.stdin.read()
    else:
        parser.print_help()
        print("\n💡  Tip: run with --demo to see it in action\n")
        return

    compressor = ContextCompressor(
        dedupe=not args.no_dedupe,
        fold_logs=not args.no_fold,
        prune_filler=not args.no_prune,
        rank_sentences=args.rank_sentences,
        keep_ratio=args.keep_ratio,
    )
    result = compressor.compress(text, content_type=args.type)

    if args.json:
        print(json.dumps(result["stats"], indent=2))
    else:
        print(BANNER)
        print_report(result)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result["compressed_text"])
        print(f"💾  Compressed text saved to: {args.output}")
    elif not args.json:
        print("COMPRESSED OUTPUT:")
        print(result["compressed_text"])


if __name__ == "__main__":
    main()
