import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Carpetas y sets de pruebas
carpetas = ['results/']
trials_por_carpeta = [(0, 31)] 


# Para almacenar los datos
datos_por_set = []
nombres_sets = []

# Lectura de los archivos .json
for k, carpeta in enumerate(carpetas):
    best_genomes_set = []  # Almacenará los resultados de un set

    # Extraer el nombre del set (parte antes del '/')
    nombre_set = carpeta.split('/')[0]
    nombres_sets.append(nombre_set)

    for j in range(trials_por_carpeta[k][0], trials_por_carpeta[k][1]):
        file_path = f"{carpeta}trial-{j+1}/output.json"
        
        # Verificar si el archivo existe
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extraer el "BestGenome"
                best_genomes = data.get('Info', {}).get('BestGenome', [])
                if best_genomes:
                    # Tomamos el último valor de fitness como el mejor por cada prueba
                    last_best_fitness = best_genomes[-1][-1] if best_genomes[-1] else None
                    if last_best_fitness is not None:
                        best_genomes_set.append(last_best_fitness)
    
    # Agregar los datos del set al conjunto general
    datos_por_set.append(best_genomes_set)

# Creación del boxplot

print("Mejor trial por set:")
for i, best_genomes_set in enumerate(datos_por_set):
    if best_genomes_set:
        mejor_trial = max(best_genomes_set)
        print(f"Set {nombres_sets[i]}: {mejor_trial} en el trial {best_genomes_set.index(mejor_trial) + 1}")

plt.figure(figsize=(10, 6))
sns.boxplot(data=datos_por_set)
nombres = ['NEAT']
plt.xticks(ticks=range(len(nombres_sets)), labels=nombres, rotation=15, ha='right')
plt.xlabel("Set de Pruebas")
plt.ylabel("Best Genome Fitness")
plt.title("Comparación de Best Genome Fitness entre Sets")
plt.grid(True)
plt.savefig('boxplot.png')
plt.show()
