#include <iostream>
#include <mpi.h>      ///< Biblioteca para programación distribuida mediante MPI.

#include "MonteCarlo.hpp"

/**
 * @ingroup MonteCarloMPI
 * @brief Función a integrar.
 *
 * Calcula el valor de la función
 *
 * \f[
 * f(x,y)=x^2+y^2
 * \f]
 *
 * utilizada como ejemplo para probar el algoritmo de integración de
 * Monte Carlo.
 *
 * @param vec Vector que representa el punto de evaluación
 * \f$(x,y)\f$.
 *
 * @return Valor de la función en el punto indicado.
 */
double fx(std::vector<double> vec){
    return vec[0] * vec[0] + vec[1] * vec[1];
}

/**
 * @brief Función principal del programa.
 *
 * Inicializa el entorno MPI, crea un objeto de la clase MonteCarlo
 * para realizar una integración bidimensional y ejecuta el algoritmo
 * utilizando distintos números de muestras.
 *
 * La función integrada es
 *
 * \f[
 * f(x,y)=x^2+y^2
 * \f]
 *
 * sobre el dominio
 *
 * \f[
 * [0,1]\times[0,1].
 * \f]
 *
 * El cálculo de la integral se distribuye entre todos los procesos MPI.
 * Sin embargo, únicamente el proceso raíz (rank 0) imprime los resultados
 * obtenidos.
 *
 * @param argc Número de argumentos de la línea de comandos.
 * @param argv Vector con los argumentos de la línea de comandos.
 *
 * @return 0 si el programa finaliza correctamente.
 */
int main(int argc, char *argv[]){

    /**
     * @brief Inicializa el entorno MPI.
     *
     * Debe ejecutarse antes de cualquier llamada a funciones de MPI.
     */
    MPI_Init(&argc, &argv);

    /// Identificador del proceso actual.
    int rank;

    /// Número total de procesos MPI.
    int size;

    // Obtención del identificador del proceso y del número total de procesos.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /**
     * @brief Objeto encargado de realizar la integración.
     *
     * Se inicializa para un problema bidimensional.
     */
    MonteCarlo dimensiones = MonteCarlo(2);

    /**
     * @brief Intervalos de integración.
     *
     * Define el dominio
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
     * @brief El proceso raíz informa la dimensión del problema.
     */
    if (rank == 0)
        std::cout << "Dimensiones: 2" << std::endl;

    /**
     * @brief Bucle de experimentación.
     *
     * Ejecuta el algoritmo utilizando un número creciente de muestras
     * aleatorias:
     *
     * 1, 10, 100, ..., 10⁹.
     *
     * Cada proceso participa en el cálculo de la integral, pero solamente
     * el proceso raíz imprime la aproximación obtenida.
     */
    for (int N = 1; N < 1000000000; N *= 10){

        /// Aproximación de la integral calculada mediante Monte Carlo.
        double I = dimensiones.integralDM(limites, N, fx);

        /// El proceso raíz muestra el resultado.
        if (rank == 0){
            std::cout << "Para " << N << " -> " << I << std::endl;
        }
    }

    /**
     * @brief Finaliza el entorno MPI.
     *
     * Libera los recursos asociados a MPI y debe ejecutarse antes de
     * terminar el programa.
     */
    MPI_Finalize();

    return 0;
}
