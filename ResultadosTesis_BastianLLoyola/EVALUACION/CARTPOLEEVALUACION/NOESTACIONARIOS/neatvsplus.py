import numpy as np
import os
from scipy.stats import kruskal

# Directorio base
base_dir = r"C:\Users\Administrador\Downloads\resultadostesis\EVALUACION\CARTPOLEEVALUACION\NOESTACIONARIOS"

# Condiciones
condiciones = ["force", "gravity", "lenght", "mass", "all"]

print("Comparación NEAT vs PLUS")
print(f"{'Condición':<10} {'H':>8} {'p-value':>10} {'Significativo':>15}")
print("-"*45)

for cond in condiciones:
    # Cargar datos
    path_neat = os.path.join(base_dir, "NEAT", cond, "retornos2_base-force-rstdp.npy")
    path_plus = os.path.join(base_dir, "PLUS", cond, "retornos2_base-force-rstdp.npy")
    
    neat = np.load(path_neat)  # shape (33, 1000)
    plus = np.load(path_plus)  # shape (33, 1000)
    
    # Promediar cada fila para tener un valor por trial
    neat_trials = neat.mean(axis=1)
    plus_trials = plus.mean(axis=1)
    
    # Kruskal-Wallis
    H, p = kruskal(neat_trials, plus_trials)
    
    # Determinar significancia
    signif = "Sí" if p < 0.05 else "No"
    
    print(f"{cond:<10} {H:8.3f} {p:10.4f} {signif:>15}")