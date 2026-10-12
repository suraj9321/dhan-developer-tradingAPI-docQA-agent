"""Send a list of questions to a running /ask endpoint and write a Markdown report.

    python -m scripts.ask_batch --out eval/results/manual_baseline.md --system baseline
"""

import argparse
import time

import httpx

QUESTIONS = [
    "Which endpoint returns the option chain for an underlying?",
    "What values can productType take when placing an order?",
    "What does error code DH-904 mean?",
    "How do I cancel an order in the Global Stocks API?",
    "What is the rate limit for order APIs per second?",
    "Which header carries the access token on REST requests?",
    "How do I set up a static IP for API access?",
    "What fields does the Get Holdings response include for Indian stocks?",
    "How do I exit all open positions at once?",
    "Does DhanHQ support trading cryptocurrency futures through the API?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--system", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = [f"# Manual check: `{args.system or 'default'}` system", ""]
    with httpx.Client(timeout=300) as client:
        for i, q in enumerate(QUESTIONS, 1):
            t0 = time.time()
            r = client.post(f"{args.url}/ask", json={"question": q, "system": args.system}).json()
            lines += [f"## {i}. {q}", "", r["answer"], ""]
            for c in r["citations"]:
                lines.append(f"- [{c['n']}] {c['citation']}  _({' > '.join([c['sub_doc'], *c['heading_path']])})_")
            lines += [
                "",
                f"`{r['latency_ms']} ms · {r['input_tokens']}+{r['output_tokens']} tokens · ${r['cost_usd']:.5f} · refused={r['refused']}`",
                "",
            ]
            print(f"{i:2d} {time.time() - t0:5.1f}s refused={r['refused']} cites={len(r['citations'])}")
    with open(args.out, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
