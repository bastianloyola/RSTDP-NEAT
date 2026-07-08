from pathlib import Path

EXPS = ["all", "force", "gravity", "length", "mass"]
TRIALS = range(1, 12)

def cortar_desde_patron(path, patron):
    if not path.exists():
        return

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lineas = f.readlines()

    for i, linea in enumerate(lineas):
        if patron in linea:
            with path.open("w", encoding="utf-8") as f:
                f.writelines(lineas[:i])
            return  # cortar solo una vez


def cortar_desde_n_esima_aparicion(path, patron, n):
    if not path.exists():
        return

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lineas = f.readlines()

    count = 0
    for i, linea in enumerate(lineas):
        if patron in linea:
            count += 1
            if count == n:
                with path.open("w", encoding="utf-8") as f:
                    f.writelines(lineas[:i])
                return


for exp in EXPS:
    for t in TRIALS:
        base = Path(exp) / "results" / f"trial-{t}"

        cortar_desde_patron(base / "evals.txt", "gen 26")
        cortar_desde_patron(base / "info.txt", "Generation: 26")
        cortar_desde_patron(base / "operadores.txt", "Generacion: 26")
        cortar_desde_n_esima_aparicion(base / "R-STDP-info.txt", "especie 0", 6)
