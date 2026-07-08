import numpy as np
import matplotlib.pyplot as plt
import os

# Directorio base
base_dir = r"C:\Users\Administrador\Downloads\resultadostesis\EVALUACION\CARTPOLEEVALUACION\NOESTACIONARIOS"

# Mapeo de condiciones: (nombre_carpeta, nombre_tesis)
condiciones_map = [
    ("force", "Fuerza"),
    ("gravity", "Gravedad"),
    ("lenght", "Longitud"),
    ("mass", "Masa"),
    ("all", "Todos")
]

# Crear figura 2x3
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes_flat = axes.flatten()

for i in range(len(axes_flat)):
    ax = axes_flat[i]
    
    # Si estamos en el índice 5 (el sexto gráfico) y no hay datos, lo ocultamos
    if i >= len(condiciones_map):
        ax.axis('off')  # <--- ESTO OCULTA EL GRÁFICO VACÍO
        continue
    
    folder, name = condiciones_map[i]
    
    try:
        # Cargar NEAT
        path_neat = os.path.join(base_dir, "NEAT", folder, "retornos2_base-force-rstdp.npy")
        neat = np.load(path_neat)

        # Cargar PLUS-RSTDP
        path_plus = os.path.join(base_dir, "PLUS", folder, "retornos2_base-force-rstdp.npy")
        plus = np.load(path_plus)

        # Concatenar episodios (1000 + 1000 = 2000)
        combined = np.concatenate([neat, plus], axis=1)

        # Calcular estadísticas
        media = np.mean(combined, axis=0)

        # Graficar línea de la media
        ax.plot(media, label="Media de Retorno", color="#1f77b4", linewidth=2.5)

        # Línea vertical para marcar el inicio de PLUS-RSTDP
        ax.axvline(x=1000, color="#d62728", linestyle="--", linewidth=2, label="Inicio Reajuste RSTDP")

        # Títulos y etiquetas
        ax.set_title(f"Escenario: {name}", fontsize=20, fontweight='bold', pad=15)
        ax.set_xlabel("Episodio", fontsize=16)
        ax.set_ylabel("Retorno", fontsize=16)
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim([0, 510]) 
        ax.grid(True, linestyle=":", alpha=0.7)
        
    except FileNotFoundError:
        ax.set_title(f"Error: No se encontró {folder}", fontsize=14, color="red")

# Ajuste de la leyenda general
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=18, frameon=True, shadow=True)

plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.25)

plt.show()