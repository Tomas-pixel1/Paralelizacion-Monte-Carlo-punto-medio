/**
 * @file MonteCarlo.cpp
 * @defgroup MonteCarloSec Implementación secuencial
 * @brief Implementación secuencial del método de integración de Monte Carlo.
 * @{
 *
 * Esta implementación contiene la versión secuencial de la clase
 * `MonteCarlo`, en la que todas las muestras aleatorias se generan y
 * procesan en un único hilo de ejecución.
 *
 * ## Descripción
 *
 * El algoritmo estima una integral múltiple mediante el método de
 * Monte Carlo. Para ello:
 * - Calcula el volumen de la región de integración.
 * - Genera N muestras aleatorias uniformemente distribuidas.
 * - Evalúa la función de usuario en cada muestra.
 * - Obtiene el promedio de las evaluaciones y lo multiplica por el
 *   volumen de integración.
 *
 * ## Características
 * - Implementación completamente secuencial.
 * - Utiliza rand() para generar números pseudoaleatorios.
 * - No emplea paralelización ni sincronización entre hilos.
 *
 * ## Complejidad
 * - Tiempo: O(N·d)
 * - Espacio: O(N·d), debido al almacenamiento temporal de las muestras.
 *
 * @date 2026
 
 * @defgroup MonteCarloMPI Implementación MPI
 * @brief Implementación del método de Monte Carlo utilizando memoria distribuida.
 * @{
 *
 * Esta implementación emplea MPI para distribuir el cálculo de la integral
 * entre múltiples procesos. Cada proceso genera una parte de las muestras
 * aleatorias, evalúa la función de integración sobre ellas y calcula una
 * suma parcial. Finalmente, todas las contribuciones se combinan mediante
 * una operación de reducción para obtener la aproximación final de la
 * integral.
 *
 * ## Descripción
 *
 * El algoritmo sigue los siguientes pasos:
 * - Calcula el volumen de la región de integración.
 * - Obtiene el número de procesos participantes y el identificador
 *   (rank) del proceso actual.
 * - Distribuye las N muestras entre los procesos disponibles,
 *   equilibrando la carga cuando N no es divisible entre el número de
 *   procesos.
 * - Cada proceso genera únicamente las muestras que le corresponden y
 *   evalúa la función sobre ellas.
 * - Las sumas parciales se combinan mediante MPI_Reduce.
 * - El proceso raíz (rank 0) calcula y devuelve la aproximación final
 *   de la integral.
 *
 * ## Estrategia de paralelización
 *
 * - Cada proceso ejecuta una parte independiente del algoritmo.
 * - La carga de trabajo se reparte de forma equilibrada.
 * - No existe memoria compartida entre procesos.
 * - La comunicación se limita a la operación colectiva MPI_Reduce.
 *
 * ## Características
 *
 * - Paralelización distribuida mediante MPI.
 * - Distribución equilibrada de las muestras.
 * - Comunicación colectiva mediante MPI_Reduce.
 * - Escalable a múltiples nodos de un clúster.
 *
 * ## Requisitos
 *
 * - Inicializar el entorno MPI mediante MPI_Init().
 * - Ejecutar con mpirun o mpiexec.
 * - Finalizar mediante MPI_Finalize().
 *
 * ## Complejidad
 *
 * - Tiempo: O((N·d)/p)
 * - Espacio: O(d)
 * - Comunicación: una reducción colectiva MPI_Reduce.
 *
 * @date 2026
 
 * @defgroup MonteCarloOMP Implementación OpenMP
 * @brief Implementación del método de Monte Carlo utilizando memoria compartida (OpenMP).
 * @{
 *
 * Esta implementación emplea OpenMP para paralelizar el cálculo de la
 * integral mediante la distribución de las muestras aleatorias entre
 * múltiples hilos de ejecución. Cada hilo genera sus propios puntos de
 * integración, evalúa la función proporcionada por el usuario y contribuye
 * al resultado final mediante una reducción paralela.
 *
 * ## Descripción
 *
 * El algoritmo realiza los siguientes pasos:
 * - Calcula el volumen de la región de integración.
 * - Crea una región paralela con OpenMP.
 * - Inicializa un generador de números aleatorios independiente para cada
 *   hilo utilizando `std::mt19937`.
 * - Genera N muestras aleatorias distribuidas uniformemente dentro del
 *   dominio de integración.
 * - Evalúa la función sobre cada muestra.
 * - Combina las contribuciones de todos los hilos mediante reducción.
 * - Multiplica el promedio de las evaluaciones por el volumen de la región.
 *
 * ## Estrategia de paralelización
 *
 * - Cada hilo usa un generador independiente.
 * - Se evita condición de carrera con variables locales.
 * - Se usa `reduction(+:cont)` para sumar resultados.
 * - `#pragma omp for` distribuye iteraciones automáticamente.
 *
 * ## Características
 *
 * - Paralelización mediante OpenMP.
 * - Generación de números aleatorios con `std::random_device` y `std::mt19937`.
 * - Reducción automática de resultados.
 * - Escalable al número de hilos disponibles.
 *
 * ## Complejidad
 *
 * - Tiempo: O((N·d)/p)
 * - Espacio: O(d) por hilo
 *
 * @date 2026
 */
 
#ifndef MONTECARLO_HPP
#define MONTECARLO_HPP

#include <vector>
#include <functional>

/**
 * @class MonteCarlo
 * @brief Implementa el método de integración numérica mediante Monte Carlo.
 *
 * Esta clase permite aproximar la integral de una función de varias
 * variables sobre una región hipercúbica definida por intervalos
 * utilizando el método de Monte Carlo.
 */
class MonteCarlo{
  public:

    /**
     * @brief Constructor de la clase.
     *
     * Inicializa el objeto especificando la dimensión del espacio
     * en el que se realizará la integración.
     *
     * @param d Dimensión del problema.
     */
    MonteCarlo(int d);

    /**
     * @brief Constructor de copia.
     *
     * Crea un nuevo objeto como copia de otro objeto MonteCarlo.
     *
     * @param obj Objeto a copiar.
     */
    MonteCarlo(const MonteCarlo &obj);

    /**
     * @brief Destructor de la clase.
     */
    ~MonteCarlo();

    /**
     * @brief Calcula una integral mediante el método de Monte Carlo.
     *
     * Genera un conjunto de muestras aleatorias dentro de los intervalos
     * especificados y estima el valor de la integral de la función.
     *
     * @param intervalos Vector de pares que define los límites inferior y
     * superior de cada dimensión.
     * @param N Número de muestras aleatorias a generar.
     * @param func Función a integrar. Recibe un vector con las coordenadas
     * del punto de evaluación y devuelve el valor de la función.
     *
     * @return Aproximación del valor de la integral.
     */
    double integral(
        const std::vector<std::pair<double, double>>& intervalos,
        int N,
        const std::function<double(std::vector<double>)>& func
    );

    /**
     * @brief Calcula una integral mediante el método de Monte Carlo paralelizado con memoria compartida.
     *
     * Genera un conjunto de muestras aleatorias dentro de los intervalos
     * especificados y estima el valor de la integral de la función.
     *
     * @param intervalos Vector de pares que define los límites inferior y
     * superior de cada dimensión.
     * @param N Número de muestras aleatorias a generar.
     * @param func Función a integrar. Recibe un vector con las coordenadas
     * del punto de evaluación y devuelve el valor de la función.
     *
     * @return Aproximación del valor de la integral.
     */
    double integralSM(
        const std::vector<std::pair<double, double>>& intervalos,
        int N,
        const std::function<double(std::vector<double>)>& func
    );

    /**
     * @brief Calcula una integral mediante el método de Monte Carlo paralelizado con memoria distribuída.
     *
     * Genera un conjunto de muestras aleatorias dentro de los intervalos
     * especificados y estima el valor de la integral de la función.
     *
     * @param intervalos Vector de pares que define los límites inferior y
     * superior de cada dimensión.
     * @param N Número de muestras aleatorias a generar.
     * @param func Función a integrar. Recibe un vector con las coordenadas
     * del punto de evaluación y devuelve el valor de la función.
     *
     * @return Aproximación del valor de la integral.
     */
    double integralDM(
        const std::vector<std::pair<double, double>>& intervalos,
        int N,
        const std::function<double(std::vector<double>)>& func
    );

  private:

    /**
     * @brief Dimensión del espacio de integración.
     */
    int dimension;

    /**
     * @brief Constructor por defecto.
     *
     * Se declara privado para impedir la creación de objetos sin
     * especificar la dimensión del problema.
     */
    MonteCarlo();

    /**
     * @brief Calcula el producto de las longitudes de los intervalos.
     *
     * Este valor corresponde al volumen de la región de integración.
     *
     * @param intervalos Vector con los límites de integración.
     *
     * @return Volumen de la región de integración.
     */
    double Productorio(const std::vector<std::pair<double, double>>& intervalos);
};

#endif
