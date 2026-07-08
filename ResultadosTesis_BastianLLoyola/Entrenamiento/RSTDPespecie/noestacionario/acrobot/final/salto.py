import re
from pathlib import Path
import matplotlib.pyplot as plt
import statistics

# =========================
# CONFIG
# =========================
BASE_PATH = Path(".")  # cartpole/final
EXPERIMENTS = ["all", "largo", "masa", "moi"]
N_TRIALS = 10
OUTPUT_DIR = Path("plots/jumps")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gen_pattern = re.compile(r"Evaluation of gen (\d+)")
fitness_pattern = re.compile(r"fitness:\s*([-+]?\d*\.?\d+)")

# =========================
# PARSEO
# =========================
def parse_experiment(exp_path):
    gen_fitness = {}

    for t in range(1, N_TRIALS + 1):
        evals_file = exp_path / "results" / f"trial-{t}" / "evals.txt"
        if not evals_file.exists():
            continue

        current_gen = None
        reading_neat = False

        with evals_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                gen_match = gen_pattern.search(line)
                if gen_match:
                    current_gen = int(gen_match.group(1))
                    reading_neat = True
                    gen_fitness.setdefault(current_gen, [])
                    continue

                if "Evaluation species" in line:
                    reading_neat = False
                    continue

                if reading_neat and current_gen is not None:
                    fit_match = fitness_pattern.search(line)
                    if fit_match:
                        gen_fitness[current_gen].append(float(fit_match.group(1)))

    return gen_fitness

# =========================
# STATS
# =========================
def stats(values):
    return {
        "best": max(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
    }

def delta(v1, v2):
    abs_delta = v2 - v1
    pct_delta = (abs_delta / abs(v1)) * 100 if v1 != 0 else 0
    return abs_delta, pct_delta

# =========================
# MAIN
# =========================
for exp in EXPERIMENTS:
    gen_fitness = parse_experiment(BASE_PATH / exp)

    if not gen_fitness:
        print(f"[WARN] No data for {exp}")
        continue

    max_gen = max(gen_fitness.keys())

    # generar automáticamente 4→5, 9→10, 14→15, ...
    checkpoints = [(g - 1, g) for g in range(5, max_gen + 1, 5)]

    metrics = {"best": [], "mean": [], "median": []}
    pct_metrics = {"best": [], "mean": [], "median": []}
    labels = []

    for g1, g2 in checkpoints:
        if g1 not in gen_fitness or g2 not in gen_fitness:
            continue
        if not gen_fitness[g1] or not gen_fitness[g2]:
            continue

        s1 = stats(gen_fitness[g1])
        s2 = stats(gen_fitness[g2])

        labels.append(f"{g1}→{g2}")

        for key in metrics:
            d_abs, d_pct = delta(s1[key], s2[key])
            metrics[key].append(d_abs)
            pct_metrics[key].append(d_pct)

    if not labels:
        print(f"[WARN] No valid checkpoints for {exp}")
        continue

    # ---- plots ----
    for key in ["best", "mean", "median"]:
        fig, ax1 = plt.subplots()

        ax1.bar(labels, metrics[key])
        ax1.set_ylabel("Δ fitness")
        ax1.set_title(f"{exp} – Δ {key} fitness at multiples of 5")
        ax1.grid(True, axis="y")

        ax2 = ax1.twinx()
        ax2.plot(labels, pct_metrics[key], marker="o")
        ax2.set_ylabel("Δ %")

        out = OUTPUT_DIR / f"delta_{key}_{exp}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[OK] Saved {out}")
