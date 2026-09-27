"""Aggregate grounding-suppression results across 5 models.

Reads selection_phase.per_scale from each model's
results/{key}_grounding_suppress/intervention_results.json
and prints a markdown table for the appendix.
"""
import json
from pathlib import Path

MODELS = [
    ("qwen",        "Qwen3-VL-8B"),
    ("llava",       "LLaVA-v1.6-7B"),
    ("llama",       "Llama-3.2-V-11B"),
    ("internvl8b",  "InternVL3-8B"),
    ("internvl14b", "InternVL3-14B"),
]

SCALES = ["baseline", "0.0", "0.25", "0.5", "0.75"]


def load_per_scale(key):
    path = Path(f"results/{key}_grounding_suppress/intervention_results.json")
    if not path.exists():
        return None
    r = json.load(open(path))
    bench = r.get("pope_adversarial") or next(iter(r.values()))
    sel = bench.get("selection_phase", {})
    base = sel.get("baseline", {})
    per = sel.get("per_scale", {})
    rows = {"baseline": (base.get("accuracy"), base.get("halluc_rate"))}
    for s in ["0.0", "0.25", "0.5", "0.75"]:
        if s in per:
            rows[s] = (per[s].get("accuracy"), per[s].get("halluc_rate"))
        else:
            rows[s] = (None, None)
    return rows


def fmt(x):
    return f"{x:.3f}" if x is not None else "  —  "


def main():
    print("\n## Grounding-pathway suppression: per-model scale sweep")
    print("(selection set, n=100; baseline = no intervention)\n")

    header = "| Model | Metric | " + " | ".join(
        ["baseline"] + [f"s={s}" for s in ["0.0", "0.25", "0.5", "0.75"]]) + " |"
    sep = "|" + "|".join(["---"] * (len(SCALES) + 2)) + "|"
    print(header)
    print(sep)

    for key, name in MODELS:
        rows = load_per_scale(key)
        if rows is None:
            print(f"| {name} | — | not found |   |   |   |   |")
            continue
        accs = [rows[s][0] for s in SCALES]
        halls = [rows[s][1] for s in SCALES]
        print(f"| {name} | acc    | "
              + " | ".join(fmt(a) for a in accs) + " |")
        print(f"|        | halluc | "
              + " | ".join(fmt(h) for h in halls) + " |")
    print()

    # Headline numbers for the paper
    print("\n## Headline: accuracy at s=0.0 vs baseline (full grounding ablation)\n")
    print("| Model | baseline acc | s=0.0 acc | Δacc |")
    print("|---|---|---|---|")
    for key, name in MODELS:
        rows = load_per_scale(key)
        if rows is None:
            continue
        b = rows["baseline"][0]
        z = rows["0.0"][0]
        if b is None or z is None:
            continue
        print(f"| {name} | {b:.3f} | {z:.3f} | {z - b:+.3f} |")
    print()


if __name__ == "__main__":
    main()
