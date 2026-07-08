import numpy as np
import os
from scipy.stats import kruskal
import scikit_posthocs as sp

# Directorio base
base_dir = r"C:\Users\Administrador\Downloads\resultadostesis\EVALUACION\ACROBOTEVALUACION\ESTACIONARIOS"

# Métodos y condiciones
metodos = ["NEAT", "PLUS", "ESPECIE", "FIJO"]
condiciones = ["est", "largo", "masa", "moi", "all"]

# Diccionario para almacenar datos promediados por trial
datos = {}

for cond in condiciones:
    datos[cond] = {}
    for metodo in metodos:
        path = os.path.join(base_dir, metodo, cond, "retornos2_base-force-rstdp.npy")
        arr = np.load(path)  # shape (33, 1000)
        # Promediamos episodios por trial → (33,)
        datos[cond][metodo] = arr.mean(axis=1)

# Análisis de significancia
for cond in condiciones:
    print(f"\nCondición: {cond}")
    data_grupos = [datos[cond][metodo] for metodo in metodos]
    
    # Prueba Kruskal-Wallis
    stat, p = kruskal(*data_grupos)
    print(f"Kruskal-Wallis H={stat:.3f}, p={p:.4f}")
    
    if p < 0.05:
        print("Diferencias significativas encontradas. Haciendo post-hoc (Dunn)...")
        # Combinar en un solo array y etiquetar los grupos
        all_values = np.concatenate(data_grupos)
        labels = np.concatenate([[m]*len(datos[cond][m]) for m in metodos])
        # Dunn test con corrección de Bonferroni
        posthoc = sp.posthoc_dunn([datos[cond][m] for m in metodos], p_adjust='bonferroni')
        posthoc.index = metodos
        posthoc.columns = metodos
        print(posthoc.round(4))
    else:
        print("No hay diferencias significativas entre métodos.")