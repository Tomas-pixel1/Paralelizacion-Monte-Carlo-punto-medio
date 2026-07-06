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
 
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <mpi.h>
#include "MonteCarlo.hpp"

/**
 * @ingroup MonteCarloSec
 * @brief Constructor de la clase MonteCarlo.
 *
 * Inicializa la dimensión del espacio sobre el cual se realizará
 * la integración mediante el método de Monte Carlo.
 *
 * @param d Dimensión del problema.
 */
MonteCarlo::MonteCarlo(int d){
    dimension = d;
}

/**
 * @ingroup MonteCarloSec
 * @brief Constructor por defecto.
 *
 * Este constructor está declarado como privado en la clase y únicamente
 * muestra un mensaje indicando que no es válido crear un objeto sin
 * especificar la dimensión.
 */
MonteCarlo::MonteCarlo(){
    std::cout << "No es posible d = 0" << std::endl;
}

/**
 * @ingroup MonteCarloSec
 * @brief Constructor de copia.
 *
 * Crea un nuevo objeto a partir de otro existente e informa de la
 * invocación del constructor de copia.
 *
 * @param obj Objeto MonteCarlo que se copiará.
 *
 * @note Actualmente la dimensión del objeto copiado no se asigna al
 * nuevo objeto; únicamente se almacena en una variable local.
 */
MonteCarlo::MonteCarlo(const MonteCarlo &obj){
    std::cout << "constructor de copia invocado" << std::endl;
    int dim = obj.dimension;
}

/**
 * @ingroup MonteCarloSec
 * @brief Destructor de la clase.
 *
 * Libera los recursos asociados al objeto e informa de la destrucción
 * del mismo.
 */
MonteCarlo::~MonteCarlo(){
    std::cout << "Destructor invocado" << std::endl;
}
/**
 * @ingroup MonteCarloSec
 * @brief Calcula el volumen de la región de integración.
 *
 * El volumen se obtiene como el producto de las longitudes de los
 * intervalos de cada dimensión.
 *
 * @param vec Vector cuyos elementos contienen los límites inferior
 * y superior de cada intervalo.
 *
 * @return Volumen de la región de integración.
 */
double MonteCarlo::Productorio(
    const std::vector<std::pair<double, double>>& vec){

    double volumen = 1.0;

    for (int i = 0; i < dimension; i++){
        volumen *= vec[i].second - vec[i].first;
    }

    return volumen;
}

/**
 * @ingroup MonteCarloSec
 * @brief Aproxima una integral múltiple mediante el método de Monte Carlo.
 *
 * El algoritmo genera N puntos aleatorios distribuidos uniformemente
 * dentro de la región de integración, evalúa la función en cada uno de
 * ellos y calcula el promedio de dichas evaluaciones multiplicado por el
 * volumen de la región.
 *
 * @param vec Vector con los intervalos de integración para cada dimensión.
 * @param N Número de muestras aleatorias.
 * @param func Función a integrar.
 *
 * @return Aproximación del valor de la integral.
 */
double MonteCarlo::integral(
    const std::vector<std::pair<double, double>>& vec,
    int N,
    const std::function<double(std::vector<double>)>& func){

    double volumen = MonteCarlo::Productorio(vec);
    double g = volumen / N;
    double cont = 0.0;

    std::vector<std::vector<double>> x;

    // Generación de muestras aleatorias y evaluación de la función.
    for (int i = 0; i < N; i++){
        std::vector<double> comp;

        for (int j = 0; j < dimension; j++){
            comp.push_back(
                rand() * (vec[j].second - vec[j].first) / RAND_MAX
                + vec[j].first
            );
        }

        x.push_back(comp);
        cont += func(x[i]);
    }

    return g * cont;
}
 
 /**
 * @ingroup MonteCarloOMP
 * @brief Aproxima la integral mediante Monte Carlo con OpenMP.
 */
double MonteCarlo::integralSM(
    const std::vector<std::pair<double, double>>& vec,
    int N,
    const std::function<double(std::vector<double>)>& func){

    double volumen = MonteCarlo::Productorio(vec);
    double g = volumen / N;
    double cont = 0.0;

#pragma omp parallel reduction(+:cont)
    {
        std::random_device rd;
        std::mt19937 gen(rd() ^ omp_get_thread_num());

        std::vector<double> comp(dimension);
        std::vector<std::uniform_real_distribution<double>> distribuciones;

        for (int j = 0; j < dimension; j++){
            distribuciones.push_back(
                std::uniform_real_distribution<double>(
                    vec[j].first,
                    vec[j].second
                )
            );
        }

#pragma omp for
        for (int i = 0; i < N; i++){
            for (int j = 0; j < dimension; j++){
                comp[j] = distribuciones[j](gen);
            }

            cont += func(comp);
        }
    }

    return g * cont;
}
 
/**
 * @ingroup MonteCarloMPI
 * @brief Aproxima una integral múltiple mediante el método de Monte Carlo
 * utilizando paralelización distribuida con MPI.
 *
 * El número total de muestras se reparte entre todos los procesos
 * disponibles. Cada proceso genera sus propios puntos aleatorios,
 * evalúa la función sobre ellos y calcula una suma parcial.
 * Finalmente, todas las contribuciones se combinan mediante
 * MPI_Reduce, obteniendo la estimación final únicamente en el
 * proceso raíz (rank 0).
 *
 * @param vec Vector que contiene los intervalos de integración para
 * cada dimensión.
 * @param N Número total de muestras aleatorias.
 * @param func Función a integrar.
 *
 * @return Aproximación del valor de la integral en el proceso raíz
 * (rank 0). Los demás procesos retornan 0.
 *
 * @note La carga de trabajo se distribuye entre los procesos de forma
 * equilibrada. Si el número de muestras no es divisible entre el número
 * de procesos, las muestras restantes se asignan a los primeros procesos.
 *
 * @note Es necesario haber inicializado previamente el entorno MPI
 * mediante MPI_Init() y finalizarlo posteriormente con MPI_Finalize().
 */
 
double MonteCarlo::integralDM(
    const std::vector<std::pair<double, double>>& vec,
    int N,
    const std::function<double(std::vector<double>)>& func){

    /// Número total de procesos MPI.
    int size;

    /// Identificador (rank) del proceso actual.
    int rank;

    // Obtención del número de procesos y del identificador del proceso.
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /// Número de muestras asignadas al proceso.
    int Nlocal = N / size;

    /// Muestras restantes cuando N no es divisible entre los procesos.
    int rest = N % size;

    // Reparto equilibrado de las muestras restantes.
    if (rest && rank < rest)
        Nlocal++;

    /// Índice inicial de las muestras asignadas al proceso.
    int start = Nlocal * rank;

    if (rest && rank >= rest)
        start += rest;

    /// Índice final (no incluido) de las muestras asignadas.
    int end = start + Nlocal;

    double volumen = MonteCarlo::Productorio(vec);

    /// Acumulador local de las evaluaciones de la función.
    double contLocal = 0.0;

    /// Punto aleatorio generado en cada iteración.
    std::vector<double> comp(vec.size());

    /**
     * @ingroup MonteCarloMPI
     * @brief Generación local de muestras aleatorias.
     *
     * Cada proceso genera únicamente las muestras correspondientes
     * a su rango de trabajo y evalúa la función sobre ellas.
     */
    for (int i = start; i < end; i++){

        for (int j = 0; j < dimension; j++){
            comp[j] = rand() *
                      (vec[j].second - vec[j].first) /
                      RAND_MAX +
                      vec[j].first;
        }

        contLocal += func(comp);
    }

    /// Suma global de todas las contribuciones.
    double cont = 0.0;
  
      /**
     * @ingroup MonteCarloMPI
     * @brief Reducción global mediante MPI.
     *
     * Las sumas parciales calculadas por cada proceso se combinan
     * utilizando la operación MPI_SUM. El resultado final queda
     * almacenado únicamente en el proceso raíz (rank 0).
     */
    MPI_Reduce(
        &contLocal,
        &cont,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    // El proceso raíz calcula la aproximación final.
    if (rank == 0)
        return volumen * cont / N;

    // Los procesos restantes no devuelven el resultado.
    return 0;
}
/** @} */
