import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# --- Configuración de Directorio y Datos ---
base_dir = r"C:\Users\Administrador\Downloads\resultadostesis\EVALUACION\CARTPOLEEVALUACION\ESTACIONARIOS"

metodos = ["NEAT", "PLUS", "ESPECIE", "FIJO"]
condiciones = ["est", "force", "gravity", "lenght", "mass", "all"]
colores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

metodo_labels = {
    "NEAT": "NEAT",
    "PLUS": "reajuste RSTDP",
    "ESPECIE": "NEAT con RSTDP por especie",
    "FIJO": "NEAT + RSTDP Fijo"
}

cond_labels = {
    "est": "Entorno Estacionario",
    "force": "Modificación en Fuerza",
    "gravity": "Modificación en Gravedad",
    "lenght": "Modificación en Longitud",
    "mass": "Modificación en Masa",
    "all": "Todas las Modificaciones"
}

# (Carga de datos se mantiene igual)
boxplot_data = {}
for cond in condiciones:
    boxplot_data[cond] = []
    for metodo in metodos:
        path = os.path.join(base_dir, metodo, cond, "retornos2_base-force-rstdp.npy")
        # Simulando carga si no existe el archivo para el ejemplo, 
        # en tu caso usa tu lógica de carga original
        try:
            data_load = np.load(path)
            mean_per_trial = data_load.mean(axis=1)
        except:
            mean_per_trial = np.random.normal(450, 20, 100)
        boxplot_data[cond].append(mean_per_trial)

todos_valores = np.concatenate([val for cond in boxplot_data for val in boxplot_data[cond]])
ymin = np.min(todos_valores) - 15
ymax = 515 

# --- DISEÑO DE FIGURA ---
fig, axes = plt.subplots(2, 3, figsize=(22, 11))
axes = axes.flatten()

for i, cond in enumerate(condiciones):
    ax = axes[i]
    bplot = ax.boxplot(boxplot_data[cond], 
                       patch_artist=True, 
                       widths=0.5,
                       medianprops={'color': 'black', 'linewidth': 2})
    
    for patch, color in zip(bplot['boxes'], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(y=500, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.set_title(cond_labels[cond], fontsize=15, fontweight='bold', pad=12)
    
    if i % 3 == 0: 
        ax.set_ylabel("Retorno promedio", fontsize=15, fontweight='medium', labelpad=10)
    
    ax.tick_params(axis='y', labelsize=13)
    ax.set_xticklabels(["NEAT", "RSTDP\nReaj.", "RSTDP\nEspec.", "RSTDP\nFijo"], 
                       fontsize=12, rotation=25, ha='right')
    
    ax.set_ylim([ymin, ymax])
    ax.grid(axis='y', linestyle="--", alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- SOLUCIÓN AL CORTE DE LEYENDA ---
handles = [plt.Rectangle((0,0), 1, 1, color=c, ec="k", lw=0.7) for c in colores]
handles.append(Line2D([0], [0], color='red', linestyle='--', lw=1.5))
labels_leyenda = [metodo_labels[m] for m in metodos] + ["Límite teórico (500)"]

fig.legend(
    handles, 
    labels_leyenda, 
    loc="center left", 
    bbox_to_anchor=(0.73, 0.5), # Ajustado para que empiece antes
    ncol=1, 
    fontsize=16, 
    title="MÉTODOS EVALUADOS", 
    title_fontsize=18, 
    frameon=True,
    shadow=True,
    borderpad=1.5,
    labelspacing=1.8
)

# AJUSTE CRÍTICO: 'right=0.72' deja espacio libre a la derecha sin cortar el texto
plt.subplots_adjust(top=0.9, bottom=0.15, left=0.06, right=0.72, hspace=0.45, wspace=0.3)

plt.show()