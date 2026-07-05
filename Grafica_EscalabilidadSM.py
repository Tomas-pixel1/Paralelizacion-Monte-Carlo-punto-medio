import numpy as np
import matplotlib.pyplot as plt
import os

def graficar_escalabilidad(archivo_entrada, archivo_salida):
    hilos = []
    tiempos = []
    
    # Lectura de los datos
    with open(archivo_entrada, "r") as archivo:
        for linea in archivo:
            try:
                h, t = linea.strip().split()
                hilos.append(int(h))
                tiempos.append(float(t))
            except ValueError:
                pass

    hilos = np.array(hilos)
    tiempos = np.array(tiempos)

    # Cálculos de escalabilidad
    tiempo_secuencial = tiempos[0]  # T_1 (tiempo con 1 hilo)
    speedup = tiempo_secuencial / tiempos
    eficiencia = speedup / hilos

    # Configuración de la figura con 2 subgráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfica de Speedup 
    ax1.plot(hilos, speedup, marker='o', linestyle='-', color='dodgerblue', 
             linewidth=2, markersize=8, label='Speedup empírico')
    
    # El speedup debería ser igual al número de hilos
    ax1.plot(hilos, hilos, marker='', linestyle='--', color='crimson', 
             linewidth=2, label='Speedup ideal')
    
    ax1.set_title("Aceleración (Speedup) vs Hilos", fontsize=14)
    ax1.set_xlabel("Número de Hilos", fontsize=12)
    ax1.set_ylabel(r"Speedup ($S = T_1 / T_N$)", fontsize=12)
    ax1.set_xticks(hilos)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=12)

    # Gráfica de Eficiencia 
    ax2.plot(hilos, eficiencia, marker='s', linestyle='-', color='forestgreen', 
             linewidth=2, markersize=8, label='Eficiencia empírica')
    
    # La eficiencia ideal es 1.0 (100%) para cualquier cantidad de hilos
    ax2.plot(hilos, np.ones_like(hilos), marker='', linestyle='--', color='crimson', 
             linewidth=2, label='Eficiencia ideal')
    
    ax2.set_title("Eficiencia vs Hilos", fontsize=14)
    ax2.set_xlabel("Número de Hilos", fontsize=12)
    ax2.set_ylabel(r"Eficiencia ($E = S / N$)", fontsize=12)
    ax2.set_xticks(hilos)
    ax2.set_ylim(0, 1.1)  # La eficiencia máxima teórica es 1.0
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend(fontsize=12)

    plt.suptitle("Análisis de Escalabilidad Fuerte - Método de Montecarlo (OpenMP)", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_resultados = os.path.join(directorio_actual, "tiempos.txt")
    graficar_escalabilidad(ruta_resultados, "Grafica_escalabilidad_memoria_compartida.png")