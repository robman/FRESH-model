"""llm_random_sampler.py

Sample “Can you give me a random number between 1 and 25?” N times across
multiple LLM APIs and local Ollama models. Computes Response Diversity Ratio.

Run:
  python llm_random_sampler.py -n 100 -m "gpt-4o-mini,gemini-1.5-flash"
"""

from __future__ import annotations

import os
import re
import csv
import argparse
import logging
from collections import Counter, defaultdict

# Optional SDKs
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

import requests
import matplotlib.pyplot as plt

PROMPT = "Can you give me a random number between 1 and 25?"
R_EXPECTED = 25  # diversity denominator

DEFAULT_MODELS = [
    {"name": "claude-3-5-sonnet-20241022", "provider": "anthropic"},
    {"name": "gpt-4o-mini",               "provider": "openai"},
    {"name": "gemini-1.5-flash",          "provider": "gemini"},
    {"name": "llama3.2",                  "provider": "ollama"},
]

# Regex finds any integer 1‑25
INTEGER_RE = re.compile(r"\b([1-9]|1\d|2[0-5])\b")

# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def extract_number(text: str) -> int | None:
    """Return *last* integer 1‑25 found in *text*, else None.

    Models often echo “…between **1 and 25**: 17”. Taking the last match
    avoids mistaking the “1” in the restatement for the actual answer.
    """
    matches = INTEGER_RE.findall(text)
    return int(matches[-1]) if matches else None

# --------------------------------------------------------------------------- #
# Provider call helpers                                                       #
# --------------------------------------------------------------------------- #
def call_openai(model: str, prompt: str) -> str:
    if not openai:
        raise RuntimeError("openai package not installed")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    return resp.choices[0].message.content.strip()

def call_anthropic(model: str, prompt: str) -> str:
    if not anthropic:
        raise RuntimeError("anthropic package not installed")
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=model,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content).strip()

def call_gemini(model: str, prompt: str) -> str:
    if not genai:
        raise RuntimeError("google-generativeai package not installed")
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    genai.configure(api_key=key)
    m = genai.GenerativeModel(model)
    resp = m.generate_content(prompt)
    return resp.text.strip()

def call_ollama(model: str, prompt: str) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()

CALL_FUNCS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "gemini": call_gemini,
    "ollama": call_ollama,
}

# --------------------------------------------------------------------------- #
# Experiment logic                                                            #
# --------------------------------------------------------------------------- #
def run_once(cfg: dict, prompt: str) -> tuple[int | None, str]:
    provider = cfg["provider"]
    func = CALL_FUNCS[provider]
    raw = func(cfg["name"], prompt)
    num = extract_number(raw)
    return num, raw

def run_experiments(models: list[dict], n_runs: int, prompt: str):
    results: defaultdict[str, list] = defaultdict(list)
    for cfg in models:
        name = cfg["name"]
        logging.info("=== %s ===", name)
        for i in range(1, n_runs + 1):
            try:
                num, raw = run_once(cfg, prompt)
                results[name].append((num, raw))
                logging.debug("%s run %d/%d -> %s", name, i, n_runs, num)
            except Exception as e:
                logging.error("%s run %d failed: %s", name, i, e)
                results[name].append((None, f"ERROR: {e}"))
    return results

# --------------------------------------------------------------------------- #
# Output helpers                                                              #
# --------------------------------------------------------------------------- #
def save_per_model_csv(results):
    for model, rows in results.items():
        fn = f"{model.replace('/', '_')}_numbers.csv"
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "number", "raw_text"])
            for idx, (num, raw) in enumerate(rows, 1):
                w.writerow([idx, num, raw])
        print(f"Wrote {fn}")

def compute_rdr(results):
    summary = []
    for model, rows in results.items():
        uniq = {n for n, _ in rows if n is not None}
        r_actual = len(uniq)
        rdr = r_actual / R_EXPECTED if R_EXPECTED else float("inf")
        summary.append((model, r_actual, rdr))
    return summary

def save_rdr_csv(summary):
    with open("rdr_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "R_actual", "RDR"])
        for model, r_act, rdr in summary:
            w.writerow([model, r_act, f"{rdr:.3f}"])
    print("Wrote rdr_summary.csv")

def plot_results(results):
    plt.figure(figsize=(10, 6))
    for model, rows in results.items():
        nums = [n for n, _ in rows if n]
        counts = Counter(nums)
        xs = list(range(1, 26))
        ys = [counts.get(x, 0) for x in xs]
        plt.plot(xs, ys, marker="o", label=model)
    plt.xticks(range(1, 26))
    plt.xlabel("Number")
    plt.ylabel("Frequency")
    plt.title("Distribution of Random Numbers per Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("llm_random_sampler_plot.png", dpi=150)
    print("Saved plot to llm_random_sampler_plot.png")

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_models_arg(arg: str | None):
    if not arg:
        return DEFAULT_MODELS
    out = []
    for m in arg.split(","):
        m = m.strip()
        provider = (
            "anthropic" if m.startswith("claude") else
            "openai"    if m.startswith("gpt")    else
            "gemini"    if m.startswith("gemini") else
            "ollama"
        )
        out.append({"name": m, "provider": provider})
    return out

def main():
    parser = argparse.ArgumentParser(description="Sample random numbers from multiple LLMs")
    parser.add_argument("-n", "--runs", type=int, default=int(os.getenv("N_RUNS", "100")),
                        help="Runs per model (default 100 or $N_RUNS)")
    parser.add_argument("-m", "--models", type=str, default=os.getenv("LLM_MODELS", ""),
                        help="Comma‑separated model list")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot creation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    models = parse_models_arg(args.models)
    logging.info("Models: %s", ", ".join(m['name'] for m in models))
    results = run_experiments(models, args.runs, PROMPT)

    save_per_model_csv(results)
    summary = compute_rdr(results)
    save_rdr_csv(summary)

    for model, r_act, rdr in summary:
        print(f"{model:25s} R_actual={r_act:2d}  RDR={rdr:.3f}")

    if not args.no_plot:
        plot_results(results)

if __name__ == "__main__":
    main()
