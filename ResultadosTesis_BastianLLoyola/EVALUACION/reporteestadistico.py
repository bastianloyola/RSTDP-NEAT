import numpy as np
import os
from scipy import stats

# --- CONFIGURACIÓN DE RUTAS ---
base_path = r"C:\Users\Administrador\Downloads\resultadostesis\EVALUACION"
bloques = {
    "CARTPOLE ESTACIONARIO": os.path.join(base_path, "CARTPOLEEVALUACION", "ESTACIONARIOS"),
    "CARTPOLE NO ESTACIONARIO": os.path.join(base_path, "CARTPOLEEVALUACION", "NOESTACIONARIOS"),
    "ACROBOT ESTACIONARIO": os.path.join(base_path, "ACROBOTEVALUACION", "ESTACIONARIOS"),
    "ACROBOT NO ESTACIONARIO": os.path.join(base_path, "ACROBOTEVALUACION", "NOESTACIONARIOS")
}

# El orden correcto de tus carpetas según los logs anteriores:
metodos = ["NEAT", "PLUS", "ESPECIE", "FIJO"]

# --- INICIO DEL ANÁLISIS ---
output_file = "REPORTE_ESTADISTICO_TESIS.txt"

# Usamos encoding='utf-8' para evitar errores de caracteres en Windows
with open(output_file, "w", encoding='utf-8') as f:
    def smart_print(texto):
        print(texto)
        f.write(texto + "\n")

    smart_print("="*90)
    smart_print(f"{'REPORTE GLOBAL DE SIGNIFICANCIA ESTADISTICA (33 TRIALS)':^90}")
    smart_print(f"{'Comparacion: Variantes RSTDP vs NEAT Base':^90}")
    smart_print("="*90)

    for nombre_bloque, ruta_bloque in bloques.items():
        smart_print(f"\n\n[[ {nombre_bloque} ]]")
        
        # Primero necesitamos saber qué condiciones existen (mirando dentro de NEAT)
        ruta_neat = os.path.join(ruta_bloque, "NEAT")
        if not os.path.exists(ruta_neat):
            smart_print(f"Error: No se encontro la carpeta NEAT en {ruta_bloque}")
            continue
            
        condiciones = [d for d in os.listdir(ruta_neat) if os.path.isdir(os.path.join(ruta_neat, d))]
        
        for cond in condiciones:
            smart_print(f"\nEscenario: {cond.upper()}")
            smart_print("-" * 85)
            smart_print(f"{'Metodo':<25} | {'Promedio':<10} | {'p-valor':<10} | {'Significancia'}")
            smart_print("-" * 85)
            
            # Cargar datos de NEAT (Control)
            path_neat = os.path.join(ruta_bloque, "NEAT", cond, "retornos2_base-force-rstdp.npy")
            try:
                data_neat = np.load(path_neat).mean(axis=1) # 33 trials
                mean_neat = np.mean(data_neat)
                smart_print(f"{'NEAT (Control)':<25} | {mean_neat:<10.2f} | {'-':<10} | {'N/A'}")
            except Exception as e:
                smart_print(f"Error cargando NEAT en {cond}: No se encontro el archivo .npy")
                continue

            # Comparar el resto de métodos contra NEAT
            for i in range(1, len(metodos)):
                m = metodos[i]
                path_m = os.path.join(ruta_bloque, m, cond, "retornos2_base-force-rstdp.npy")
                
                try:
                    data_m = np.load(path_m).mean(axis=1)
                    mean_m = np.mean(data_m)
                    
                    # Prueba Mann-Whitney U
                    stat, p_val = stats.mannwhitneyu(data_neat, data_m, alternative='two-sided')
                    
                    # Etiquetado
                    if p_val < 0.001: sig = "*** (Muy Alta)"
                    elif p_val < 0.01: sig = "** (Alta)"
                    elif p_val < 0.05: sig = "* (Significativa)"
                    else: sig = "ns (No significativa)"
                    
                    if p_val < 0.001:
                        p_text = "< 0.001"
                    elif p_val < 0.01:
                        p_text = f"{p_val:.4f}"
                    elif p_val < 0.05:
                        p_text = f"{p_val:.4f}"
                    else:
                        p_text = f"{p_val:.4f}"

                    # Formato en notación científica
                    p_text = f"{p_val:.2e}"

                    smart_print(f"{m:<25} | {mean_m:<10.2f} | {p_text:<12} | {sig}")
                
                except Exception as e:
                    smart_print(f"{m:<25} | Error: Archivo no encontrado en {cond}")

    smart_print("\n" + "="*90)
    smart_print("FIN DEL REPORTE")

print(f"\nAnalisis completado. Resultados guardados en: {output_file}")