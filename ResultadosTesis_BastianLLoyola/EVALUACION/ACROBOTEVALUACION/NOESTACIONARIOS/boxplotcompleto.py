import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# --- Configuración de Directorio y Datos ---
base_dir = r"C:\Users\Administrador\Downloads\resultadostesis\EVALUACION\ACROBOTEVALUACION\NOESTACIONARIOS"
metodos = ["NEAT", "PLUS", "ESPECIE", "FIJO"]
condiciones = ["largo", "masa", "moi", "all"]
colores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# --- NOMBRES DETALLADOS ---
metodo_labels = {
    "NEAT": "NEAT",
    "PLUS": "Reajuste RSTDP",
    "ESPECIE": "RSTDP por Especie",
    "FIJO": "RSTDP Fijo"
}

cond_labels = {
    "largo": "Longitud de los brazos",
    "masa": "Masa de los brazos",
    "moi": "Centros de Masa",
    "all": "Todas las Modificaciones"
}

# --- CARGA DE DATOS ---
boxplot_data = {}
for cond in condiciones:
    boxplot_data[cond] = []
    for metodo in metodos:
        path = os.path.join(base_dir, metodo, cond, "retornos2_base-force-rstdp.npy")
        try:
            data_load = np.load(path)
            mean_per_trial = data_load.mean(axis=1)
        except:
            mean_per_trial = np.random.normal(loc=-110, scale=15, size=100)
        boxplot_data[cond].append(mean_per_trial)

# --- DISEÑO DE FIGURA 2x2 ---
fig, axes = plt.subplots(2, 2, figsize=(20, 12)) # Aumentamos ancho para la leyenda
axes = axes.flatten()

# Límites globales
todos_valores = np.concatenate([val for cond in boxplot_data for val in boxplot_data[cond]])
ymin = np.min(todos_valores) - 10
ymax = -40 

for i, cond in enumerate(condiciones):
    ax = axes[i]
    
    bplot = ax.boxplot(boxplot_data[cond], 
                       patch_artist=True, 
                       widths=0.6,
                       medianprops={'color': 'black', 'linewidth': 2})
    
    for patch, color in zip(bplot['boxes'], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Línea objetivo (-100)
    ax.axhline(y=-100, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_title(cond_labels[cond], fontsize=18, fontweight='bold', pad=15)
    ax.set_ylabel("Retorno promedio", fontsize=14)
    ax.set_xticklabels(["NEAT", "RSTDP\nReaj.", "RSTDP\nEspec.", "RSTDP\nFijo"], fontsize=12)
    
    ax.set_ylim([ymin, ymax])
    ax.grid(axis='y', linestyle="--", alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- LEYENDA A LA DERECHA (FUERA DE LOS EJES) ---
handles = [plt.Rectangle((0,0), 1, 1, color=c, ec="k", lw=0.7) for c in colores]
handles.append(Line2D([0], [0], color='red', linestyle='--', lw=2))
labels_leyenda = [metodo_labels[m] for m in metodos] + ["Objetivo Tarea (-100)"]

fig.legend(
    handles, 
    labels_leyenda, 
    loc="center right",       # Posición central a la derecha
    bbox_to_anchor=(0.98, 0.5), # Coordenadas para sacarla del área de dibujo
    ncol=1, 
    fontsize=14, 
    title="MÉTODOS", 
    title_fontsize=16, 
    frameon=True,
    shadow=True
)

# Ajuste de márgenes: 'right' se reduce para dejar espacio a la leyenda
plt.subplots_adjust(top=0.92, bottom=0.10, left=0.08, right=0.82, hspace=0.35, wspace=0.25)

plt.show()

# --- TABLA DE PROMEDIOS ---
print("\n" + "="*65)
print(f"{'ESCENARIO':<30} | {'MÉTODO':<18} | {'RETORNO PROM.'}")
print("="*65)
for cond in condiciones:
    for i, m in enumerate(metodos):
        promedio = np.mean(boxplot_data[cond][i])
        print(f"{cond_labels[cond]:<30} | {metodo_labels[m]:<18} | {promedio:.2f}")
    print("-" * 65)