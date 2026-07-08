import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# --- Configuración de Datos (Rutas originales) ---
base_dir = r"C:\Users\Administrador\Downloads\resultadostesis\EVALUACION\CARTPOLEEVALUACION\NOESTACIONARIOS"
metodos = ["NEAT", "PLUS", "ESPECIE", "FIJO"]
condiciones = ["force", "gravity", "lenght", "mass", "all"]
colores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

metodo_labels = {
    "NEAT": "NEAT",
    "PLUS": "reajuste RSTDP",
    "ESPECIE": "NEAT con RSTDP por especie",
    "FIJO": "NEAT + RSTDP Fijo"
}

cond_labels = {
    "force": "Modificación en Fuerza",
    "gravity": "Modificación en Gravedad",
    "lenght": "Modificación en Longitud del Poste",
    "mass": "Modificación en Masa del Poste",
    "all": "Todas las Modificaciones"
}

# --- CARGA DE DATOS REALES ---
boxplot_data = {}
for cond in condiciones:
    boxplot_data[cond] = []
    for metodo in metodos:
        path = os.path.join(base_dir, metodo, cond, "retornos2_base-force-rstdp.npy")
        data_load = np.load(path)
        # Promedio por trial (asumiendo axis=1 son los pasos/episodios)
        mean_per_trial = data_load.mean(axis=1)
        boxplot_data[cond].append(mean_per_trial)

# Límites de los ejes
todos_valores = np.concatenate([val for cond in boxplot_data for val in boxplot_data[cond]])
ymin = np.min(todos_valores) - 20
# Forzamos el techo visual un poco arriba de 500 para que se vea la línea de límite
ymax = 515 

# --- DISEÑO CON GRIDSPEC ---
fig = plt.figure(figsize=(22, 13))
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.45)

posiciones = [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 2]]
axes = []

for i, cond in enumerate(condiciones):
    ax = fig.add_subplot(posiciones[i])
    axes.append(ax)
    
    bplot = ax.boxplot(boxplot_data[cond], 
                       patch_artist=True, 
                       widths=0.5,
                       medianprops={'color': 'black', 'linewidth': 2},
                       showfliers=True) # Muestra los puntos atípicos
    
    for patch, color in zip(bplot['boxes'], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # --- LÍNEA DE LÍMITE TEÓRICO (500) ---
    ax.axhline(y=500, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label="Máximo Teórico")

    ax.set_title(cond_labels[cond], fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel("Retorno promedio", fontsize=14, fontweight='medium')
    ax.tick_params(axis='y', labelsize=12)
    
    ax.set_xticklabels(["NEAT", "RSTDP\nReaj.", "RSTDP\nEspec.", "RSTDP\nFijo"], 
                       fontsize=12, rotation=25, ha='right')
    
    ax.set_ylim([ymin, ymax])
    ax.grid(axis='y', linestyle="--", alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- LEYENDA CENTRAL ---
handles = [plt.Rectangle((0,0), 1, 1, color=c, ec="k", lw=0.7) for c in colores]
# Añadimos la línea de 500 a la leyenda
from matplotlib.lines import Line2D
line_handle = Line2D([0], [0], color='red', linestyle='--', linewidth=1.5)
handles.append(line_handle)
labels_con_limite = [metodo_labels[m] for m in metodos] + ["Límite CartPole (500)"]

ax_leg = fig.add_subplot(gs[1, 1])
ax_leg.axis('off')

ax_leg.legend(
    handles, 
    labels_con_limite, 
    loc="center", 
    ncol=1, 
    fontsize=15, 
    title="MÉTODOS Y REFERENCIAS", 
    title_fontsize=17, 
    frameon=True,
    shadow=True,
    borderpad=1.8,
    labelspacing=1.8
)

plt.subplots_adjust(top=0.92, bottom=0.12, left=0.08, right=0.95)
plt.show()

# --- IMPRESIÓN DE TABLA ---
print("\n" + "-" * 60)
print(f"{'CONDICIÓN':<20} | {'MÉTODO':<15} | {'PROMEDIO TOTAL'}")
print("-" * 60)
for cond in condiciones:
    for i, metodo in enumerate(metodos):
        valor_final = np.mean(boxplot_data[cond][i])
        print(f"{cond:<20} | {metodo:<15} | {valor_final:.2f}")
    print("-" * 60)