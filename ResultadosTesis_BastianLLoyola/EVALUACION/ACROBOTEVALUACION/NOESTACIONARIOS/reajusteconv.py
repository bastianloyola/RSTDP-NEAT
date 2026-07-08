import numpy as np
import matplotlib.pyplot as plt
import os

# Directorio base
base_dir = r"C:\Users\Administrador\Downloads\resultadostesis\EVALUACION\ACROBOTEVALUACION\NOESTACIONARIOS"

# Mapeo de condiciones (4 escenarios = Distribución 2x2 perfecta)
condiciones_map = [
    ("largo", "Longitud de los Brazos"),
    ("masa", "Masa de los Brazos"),
    ("moi", "Centro de Masa de los Brazos"),
    ("all", "Todos")
]

# Crear figura 2x2
fig, axes = plt.subplots(2, 2, figsize=(18, 12)) # Ajustamos un poco el figsize para 2x2
axes = axes.flatten()

for i, (folder, name) in enumerate(condiciones_map):
    ax = axes[i]
    
    try:
        # Rutas de carga
        path_neat = os.path.join(base_dir, "NEAT", folder, "retornos2_base-force-rstdp.npy")
        path_plus = os.path.join(base_dir, "PLUS", folder, "retornos2_base-force-rstdp.npy")
        
        neat = np.load(path_neat)
        plus = np.load(path_plus)

        # Concatenación de fases (1000 + 1000)
        combined = np.concatenate([neat, plus], axis=1)
        media = np.mean(combined, axis=0)

        # Gráfica de la media
        ax.plot(media, label="Media de Retorno", color="#1f77b4", linewidth=2.5)
        
        # Marca de transición a fase PLUS (RSTDP)
        ax.axvline(x=1000, color="#d62728", linestyle="--", linewidth=2, label="Inicio Reajuste RSTDP")

        # Configuración estética
        ax.set_title(f"Escenario: {name}", fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel("Episodio", fontsize=16)
        ax.set_ylabel("Retorno", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Rango para Acrobot (Valores negativos)
        ax.set_ylim([-175, -50]) 
        ax.grid(True, linestyle=":", alpha=0.7)
        
    except FileNotFoundError:
        ax.set_title(f"Error: No se encontró {folder}", fontsize=14, color="red")

# Leyenda unificada (Posicionada arriba de los gráficos)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=18, frameon=True, shadow=True)

plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.20) # Reducimos un poco el wspace para 2x2

plt.show()