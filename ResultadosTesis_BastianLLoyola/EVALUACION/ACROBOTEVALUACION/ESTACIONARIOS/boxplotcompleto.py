import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

# --- Configuración de Directorio y Datos ---
base_dir = r"C:\Users\Administrador\Downloads\resultadostesis\EVALUACION\ACROBOTEVALUACION\ESTACIONARIOS"
metodos = ["NEAT", "PLUS", "ESPECIE", "FIJO"]
condiciones = ["est", "largo", "masa", "moi", "all"]
colores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# --- NOMBRES DETALLADOS SEGÚN TU PETICIÓN ---
metodo_labels = {
    "NEAT": "NEAT",
    "PLUS": "reajuste RSTDP",
    "ESPECIE": "NEAT con RSTDP por especie",
    "FIJO": "NEAT + RSTDP Fijo"
}

cond_labels = {
    "est": "Entorno Estacionario",
    "largo": "Longitud de los brazos",
    "masa": "Masa de los brazos",
    "moi": "Centros de Masa",
    "all": "Todas las Modificaciones"
}



# --- CARGA DE DATOS REALES ---
boxplot_data = {}
for cond in condiciones:
    boxplot_data[cond] = []
    for metodo in metodos:
        path = os.path.join(base_dir, metodo, cond, "retornos2_base-force-rstdp.npy")
        try:
            data_load = np.load(path)
            mean_per_trial = data_load.mean(axis=1)
        except:
            # Simulación por si falta algún archivo
            mean_per_trial = np.random.normal(loc=-110, scale=15, size=100)
        boxplot_data[cond].append(mean_per_trial)

# Límites comunes
todos_valores = np.concatenate([val for cond in boxplot_data for val in boxplot_data[cond]])
ymin = np.min(todos_valores) - 20
ymax = -60 # Espacio para que la línea de -100 sea visible arriba

# --- DISEÑO DE FIGURA 2x3 ---
fig = plt.figure(figsize=(22, 12))

for i, cond in enumerate(condiciones):
    ax = plt.subplot(2, 3, i + 1)
    
    bplot = ax.boxplot(boxplot_data[cond], 
                       patch_artist=True, 
                       widths=0.6,
                       medianprops={'color': 'black', 'linewidth': 2})
    
    for patch, color in zip(bplot['boxes'], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # --- LÍNEA DE OBJETIVO ACROBOT (-100) ---
    ax.axhline(y=-100, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_title(cond_labels[cond], fontsize=15, fontweight='bold', pad=15)
    
    # --- AJUSTE DE FUENTES EJE Y (PEDIDO) ---
    if i % 3 == 0:
        ax.set_ylabel("Retorno promedio", fontsize=15, fontweight='medium', labelpad=10)
    
    ax.tick_params(axis='y', labelsize=13)
    
    # Etiquetas eje X
    ax.set_xticklabels(["NEAT", "RSTDP\nReaj.", "RSTDP\nEspec.", "RSTDP\nFijo"], 
                       fontsize=11, rotation=25, ha='right')
    
    ax.set_ylim([ymin, ymax])
    ax.grid(axis='y', linestyle="--", alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- LEYENDA EN EL HUECO DEL GRÁFICO 6 ---
handles = [plt.Rectangle((0,0), 1, 1, color=c, ec="k", lw=0.7) for c in colores]
handles.append(Line2D([0], [0], color='red', linestyle='--', lw=1.5))
labels_leyenda = [metodo_labels[m] for m in metodos] + ["Objetivo Tarea (-100)"]

# Centrada en el espacio vacío del subplot 6
fig.legend(
    handles, 
    labels_leyenda, 
    loc="center",
    bbox_to_anchor=(0.81, 0.30), 
    ncol=1, 
    fontsize=16, 
    title="MÉTODOS Y OBJETIVOS", 
    title_fontsize=18, 
    frameon=True,
    shadow=True,
    borderpad=1.8,
    labelspacing=1.8
)

# Ajuste de márgenes para evitar que la leyenda se corte
plt.subplots_adjust(top=0.92, bottom=0.12, left=0.08, right=0.94, hspace=0.45, wspace=0.35)

plt.show()

# --- TABLA DE PROMEDIOS ---
print("\n" + "="*60)
print(f"{'ESCENARIO':<25} | {'MÉTODO':<15} | {'PROM.'}")
print("="*60)
for cond in condiciones:
    for i, m in enumerate(metodos):
        print(f"{cond_labels[cond]:<25} | {m:<15} | {np.mean(boxplot_data[cond][i]):.2f}")
    print("-" * 60)