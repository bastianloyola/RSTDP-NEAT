import re
from pathlib import Path
import matplotlib.pyplot as plt
import math

# =========================
# CONFIG
# =========================
BASE_PATH = Path(".")  # cartpole/final
EXPERIMENTS = ["all", "largo", "masa", "moi"]
N_TRIALS = 11
OUTPUT_DIR = Path("plots")

OUTPUT_DIR.mkdir(exist_ok=True)

gen_pattern = re.compile(r"Evaluation of gen (\d+)")
fitness_pattern = re.compile(r"fitness:\s*([-+]?\d*\.?\d+)")

# =========================
# PARSEO
# =========================
def parse_experiment(exp_path):
    """
    dict[generation] = list of fitness (todos los trials, solo NEAT)
    """
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

                # Bloques R-STDP → ignorar
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
def mean_std(values):
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)

# =========================
# PLOTS
# =========================
for exp in EXPERIMENTS:
    exp_path = BASE_PATH / exp
    gen_fitness = parse_experiment(exp_path)


    gens = sorted(gen_fitness.keys())
    means = []
    stds = []

    for g in gens:
        if len(gen_fitness[g]) > 0:
            m, s = mean_std(sorted(gen_fitness[g])[:15])
            means.append(m)
            stds.append(s)

    gens = gens[:len(means)]

    plt.figure()
    plt.plot(gens, means)
    plt.fill_between(
        gens,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.25
    )

    plt.xlabel("Generation")
    plt.ylabel("Mean fitness (± std)")
    plt.title(f"Convergence – {exp}")
    plt.grid(True)

    out_file = OUTPUT_DIR / f"convergence_{exp}_15.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved {out_file}")
