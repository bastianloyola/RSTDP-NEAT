import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
BASE_PATH = Path(".")          # cartpole/final
EXPERIMENTS = ['all', 'force', 'length', 'mass']

N_TRIALS = 11

OUTPUT_DIR = Path("plots/jumps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# patrones
gen_pattern = re.compile(r"Evaluation of gen (\d+)")
fitness_pattern = re.compile(r"fitness:\s*([-+]?\d*\.?\d+)")

# =========================
# PARSEO
# =========================
def parse_experiment(exp_path):
    """
    Retorna:
    trial_gen_mean[trial][gen] = mean fitness of genomas NEAT
    """
    trial_gen_mean = {}

    for t in range(1, N_TRIALS + 1):
        evals_file = exp_path / "results" / f"trial-{t}" / "evals.txt"
        if not evals_file.exists():
            continue

        gen_values = {}
        current_gen = None
        reading_neat = False

        with evals_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                g = gen_pattern.search(line)
                if g:
                    current_gen = int(g.group(1))
                    gen_values.setdefault(current_gen, [])
                    reading_neat = True
                    continue

                # corta al entrar a evaluación R-STDP
                if "Evaluation species" in line:
                    reading_neat = False
                    continue

                if reading_neat and current_gen is not None:
                    m = fitness_pattern.search(line)
                    if m:
                        gen_values[current_gen].append(float(m.group(1)))

        # promedio por generación
        trial_gen_mean[t] = {
            g: np.mean(v)
            for g, v in gen_values.items()
            if len(v) > 0
        }

    return trial_gen_mean

# =========================
# MAIN
# =========================
for exp in EXPERIMENTS:
    exp_path = BASE_PATH / exp
    if not exp_path.exists():
        continue

    trial_gen = parse_experiment(exp_path)
    if not trial_gen:
        print(f"[WARN] No data for {exp}")
        continue

    # generaciones disponibles
    all_gens = sorted(
        set(g for t in trial_gen.values() for g in t.keys())
    )

    max_gen = max(all_gens)

    # transiciones (k*5 - 1) → (k*5)
    checkpoints = [(g - 1, g) for g in range(5, max_gen + 1, 5)]

    labels = []
    mean_deltas = []
    std_deltas = []

    for g1, g2 in checkpoints:
        deltas = []

        for t, gen_map in trial_gen.items():
            if g1 in gen_map and g2 in gen_map:
                deltas.append(gen_map[g2] - gen_map[g1])

        if len(deltas) < 2:
            continue

        labels.append(f"{g1}→{g2}")
        mean_deltas.append(np.mean(deltas))
        std_deltas.append(np.std(deltas, ddof=1))

    if not labels:
        print(f"[WARN] No valid transitions for {exp}")
        continue

    # =========================
    # PLOT
    # =========================
    x = np.arange(len(labels))
    mean_deltas = np.array(mean_deltas)
    std_deltas = np.array(std_deltas)

    plt.figure(figsize=(11, 5))
    plt.plot(x, mean_deltas, marker="o", label="Mean Δ fitness")
    plt.fill_between(
        x,
        mean_deltas - std_deltas,
        mean_deltas + std_deltas,
        alpha=0.3,
        label="±1 std"
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, labels, rotation=45)
    plt.xlabel("Generation transition")
    plt.ylabel("Δ mean fitness")
    plt.title(f"{exp} – Fitness jumps at multiples of 5")
    plt.grid(True)
    plt.legend()

    out = OUTPUT_DIR / f"delta_fitness_{exp}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved {out}")
