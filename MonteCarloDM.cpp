#include <cstdlib>
#include <iostream>
#include <vector>
#include <mpi.h>   ///< Biblioteca para programación distribuida con MPI.

#include "MonteCarlo.hpp"

/**
 * @brief Constructor de la clase MonteCarlo.
 *
 * Inicializa la dimensión del espacio sobre el que se realizará
 * la integración mediante el método de Monte Carlo.
 *
 * @param d Dimensión del problema.
 */
MonteCarlo::MonteCarlo(int d){
    dimension = d;
}

/**
 * @brief Constructor por defecto.
 *
 * Este constructor está declarado como privado y evita la creación
 * de objetos sin especificar la dimensión del problema.
 */
MonteCarlo::MonteCarlo(){
    std::cout << "No es posible d = 0" << std::endl;
}

/**
 * @brief Constructor de copia.
 *
 * Crea un nuevo objeto como copia de otro objeto MonteCarlo.
 *
 * @param obj Objeto que se copiará.
 *
 * @note Actualmente la dimensión del objeto copiado se almacena
 * únicamente en una variable local y no se asigna al nuevo objeto.
 */
MonteCarlo::MonteCarlo(const MonteCarlo &obj){
    std::cout << "constructor de copia invocado" << std::endl;
    int dim = obj.dimension;
}

/**
 * @brief Destructor de la clase.
 *
 * Muestra un mensaje indicando la destrucción del objeto.
 */
MonteCarlo::~MonteCarlo(){
    std::cout << "Destructor invocado" << std::endl;
}

/**
 * @brief Calcula el volumen de la región de integración.
 *
 * El volumen corresponde al producto de las longitudes de los
 * intervalos de integración en cada dimensión.
 *
 * @param vec Vector con los intervalos de integración.
 *
 * @return Volumen de la región de integración.
 */
double MonteCarlo::Productorio(const std::vector<std::pair<double, double>>& vec){
    double volumen = 1.0;

    for (int i = 0; i < dimension; i++){
        volumen *= vec[i].second - vec[i].first;
    }

    return volumen;
}

/**
 * @brief Aproxima una integral múltiple mediante el método de Monte Carlo
 * utilizando paralelización distribuida con MPI.
 *
 * El número total de muestras se reparte entre todos los procesos
 * disponibles. Cada proceso genera sus propios puntos aleatorios,
 * evalúa la función sobre ellos y calcula una suma parcial.
 * Finalmente, todas las contribuciones se combinan mediante
 * `MPI_Reduce`, obteniendo la estimación final únicamente en el
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
 * mediante `MPI_Init()` y finalizarlo posteriormente con `MPI_Finalize()`.
 */
double MonteCarlo::integral(
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
     * @brief Generación local de muestras aleatorias.
     *
     * Cada proceso genera únicamente las muestras correspondientes a su
     * rango de trabajo y evalúa la función sobre ellas.
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

