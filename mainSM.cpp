#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>      ///< Biblioteca para programación paralela con OpenMP.

#include "MonteCarlo.hpp"

/**
 * @brief Función a integrar.
 *
 * Calcula el valor de la función
 *
 * \f[
 * f(x,y)=x^2+y^2
 * \f]
 *
 * utilizada para evaluar el desempeño del algoritmo de integración
 * de Monte Carlo.
 *
 * @param vec Vector que representa el punto \f$(x,y)\f$ donde se
 * evalúa la función.
 *
 * @return Valor de la función en el punto especificado.
 */
double fx(std::vector<double> vec){
    return vec[0] * vec[0] + vec[1] * vec[1];
}

/**
 * @brief Función principal del programa.
 *
 * Evalúa la escalabilidad de la implementación paralela del método de
 * Monte Carlo utilizando OpenMP.
 *
 * El programa ejecuta la integración de la función
 *
 * \f[
 * f(x,y)=x^2+y^2
 * \f]
 *
 * sobre el dominio
 *
 * \f[
 * [0,1]\times[0,1]
 * \f]
 *
 * utilizando un número fijo de muestras. Posteriormente repite el cálculo
 * incrementando el número de hilos desde 1 hasta el máximo disponible en
 * el sistema, midiendo el tiempo de ejecución en cada caso.
 *
 * Los tiempos obtenidos se almacenan en un archivo denominado
 * **tiempos.txt**, el cual puede emplearse posteriormente para realizar
 * análisis de escalabilidad y calcular métricas como el *speedup* o la
 * eficiencia paralela.
 *
 * @return 0 si el programa finaliza correctamente.
 */
int main(){

    /**
     * @brief Objeto encargado de realizar la integración.
     *
     * Se inicializa para un problema de dos dimensiones.
     */
    MonteCarlo dimensiones = MonteCarlo(2);

    /**
     * @brief Intervalos de integración.
     *
     * Define el dominio bidimensional
     *
     * \f[
     * [0,1]\times[0,1].
     * \f]
     */
    std::vector<std::pair<double, double>> limites = {
        {0.0, 1.0},
        {0.0, 1.0}
    };

    /**
     * @brief Número de muestras utilizadas en cada experimento.
     *
     * Se mantiene constante para comparar únicamente el efecto del
     * número de hilos sobre el tiempo de ejecución.
     */
    int N = 100000000;

    /**
     * @brief Número máximo de hilos disponibles.
     *
     * Se obtiene automáticamente mediante OpenMP.
     */
    int max_hilos = omp_get_max_threads();

    /**
     * @brief Archivo donde se almacenan los tiempos de ejecución.
     *
     * Cada línea contiene:
     * - Número de hilos.
     * - Tiempo de ejecución (segundos).
     */
    std::ofstream archivo("tiempos.txt");

    std::cout << "Iniciando escalabilidad con N = " << N << std::endl;
    std::cout << "Hilos disponibles: " << max_hilos << std::endl;

    /**
     * @brief Experimento de escalabilidad.
     *
     * Para cada cantidad de hilos:
     * - Se configura OpenMP.
     * - Se ejecuta la integración.
     * - Se mide el tiempo de ejecución.
     * - Se imprime el resultado por pantalla.
     * - Se almacena el tiempo en un archivo.
     */
    for (int hilos = 1; hilos <= max_hilos; hilos++){

        /// Configura el número de hilos que utilizará OpenMP.
        omp_set_num_threads(hilos);

        /// Tiempo inicial del experimento.
        double inicio = omp_get_wtime();

        /// Aproximación de la integral mediante Monte Carlo.
        double I = dimensiones.integral(limites, N, fx);

        /// Tiempo final del experimento.
        double fin = omp_get_wtime();

        /// Tiempo total de ejecución.
        double tiempo_ejecucion = fin - inicio;

        std::cout << "Hilos: " << hilos
                  << " | Tiempo: " << tiempo_ejecucion
                  << " s | Integral: " << I
                  << std::endl;

        // Guarda los datos para análisis posteriores.
        archivo << hilos << " "
                << tiempo_ejecucion
                << "\n";
    }

    /**
     * @brief Cierra el archivo de salida.
     */
    archivo.close();

    return 0;
}
