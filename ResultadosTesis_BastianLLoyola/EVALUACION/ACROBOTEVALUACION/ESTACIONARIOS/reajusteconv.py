import numpy as np
import matplotlib.pyplot as plt
import os

# Directorio base
base_dir = r"C:\Users\Administrador\Downloads\resultadostesis\EVALUACION\ACROBOTEVALUACION\ESTACIONARIOS"

# Mapeo de condiciones: (nombre_carpeta, nombre_tesis)
condiciones_map = [
    ("est", "Estacionario"),
    ("largo", "Longitud de los Brazos"),
    ("masa", "Masa de los Brazos"),
    ("moi", "Centros de Masa"),
    ("all", "Todos")
]

# Crear figura 2x3 (6 espacios en total)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

for i in range(len(axes)):
    ax = axes[i]
    
    # Si el índice i es mayor o igual al número de condiciones, ocultamos el eje
    if i >= len(condiciones_map):
        ax.axis('off')  # <--- ESTO OCULTA EL GRÁFICO ADICIONAL
        continue
        
    folder, name = condiciones_map[i]
    
    try:
        # Cargar datos
        path_neat = os.path.join(base_dir, "NEAT", folder, "retornos2_base-force-rstdp.npy")
        path_plus = os.path.join(base_dir, "PLUS", folder, "retornos2_base-force-rstdp.npy")
        
        neat = np.load(path_neat)
        plus = np.load(path_plus)

        # Concatenar episodios
        combined = np.concatenate([neat, plus], axis=1)
        media = np.mean(combined, axis=0)

        # Graficar
        ax.plot(media, label="Media de Retorno", color="#1f77b4", linewidth=2.5)
        ax.axvline(x=1000, color="#d62728", linestyle="--", linewidth=2, label="Inicio Reajuste RSTDP")

        # Formato de Acrobot (Nótese el ajuste de ylim para valores negativos)
        ax.set_title(f"Escenario: {name}", fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel("Episodio", fontsize=16)
        ax.set_ylabel("Retorno", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        ax.set_ylim([-175, -50]) 
        ax.grid(True, linestyle=":", alpha=0.7)
        
    except FileNotFoundError:
        ax.set_title(f"Error: No se encontró {folder}", fontsize=14, color="red")

# Ajuste de la leyenda general
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=18, frameon=True, shadow=True)

plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.25)

plt.show()