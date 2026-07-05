#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <omp.h>   ///< Biblioteca para programación paralela con OpenMP.

#include "MonteCarlo.hpp"

/**
 * @brief Constructor de la clase MonteCarlo.
 *
 * Inicializa la dimensión del espacio sobre el que se realizará la
 * integración mediante el método de Monte Carlo.
 *
 * @param d Dimensión del dominio de integración.
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
 * Crea un nuevo objeto a partir de otro objeto MonteCarlo.
 *
 * @param obj Objeto que se copiará.
 *
 * @note Actualmente la dimensión del objeto copiado únicamente se almacena
 * en una variable local y no se asigna al nuevo objeto.
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
 * El volumen corresponde al producto de las longitudes de los intervalos
 * de integración en cada dimensión.
 *
 * @param vec Vector que contiene los intervalos de integración.
 * Cada elemento representa un par (límite inferior, límite superior).
 *
 * @return Volumen total de la región de integración.
 */
double MonteCarlo::Productorio(const std::vector<std::pair<double, double>>& vec){
    double volumen = 1.0;

    for (int i = 0; i < dimension; i++){
        volumen *= vec[i].second - vec[i].first;
    }

    return volumen;
}

/**
 * @brief Aproxima una integral múltiple utilizando el método de Monte Carlo
 *        con paralelización mediante OpenMP.
 *
 * Se generan @p N muestras aleatorias uniformemente distribuidas dentro
 * de la región de integración. Cada hilo genera sus propias muestras,
 * evalúa la función sobre ellas y acumula el resultado mediante una
 * reducción paralela.
 *
 * Para evitar condiciones de carrera:
 * - Cada hilo utiliza un generador de números aleatorios independiente
 *   basado en `std::mt19937`.
 * - Se emplea la cláusula `reduction(+:cont)` para combinar las sumas
 *   parciales de todos los hilos al finalizar la región paralela.
 * - Cada hilo dispone de un vector local para almacenar el punto generado,
 *   evitando accesos concurrentes a memoria compartida.
 *
 * @param vec Vector con los intervalos de integración para cada dimensión.
 * @param N Número de muestras aleatorias.
 * @param func Función a integrar. Recibe un vector de coordenadas y devuelve
 * el valor de la función en dicho punto.
 *
 * @return Aproximación del valor de la integral.
 *
 * @note La calidad de la aproximación mejora al aumentar el número de
 * muestras aleatorias.
 *
 * @note La implementación utiliza OpenMP para distribuir las iteraciones
 * del algoritmo entre los hilos disponibles.
 */
double MonteCarlo::integral(
    const std::vector<std::pair<double, double>>& vec,
    int N,
    const std::function<double(std::vector<double>)>& func){

    double volumen = MonteCarlo::Productorio(vec);
    double g = volumen / N;
    double cont = 0.0;

    /**
     * @brief Región paralela de OpenMP.
     *
     * Cada hilo:
     * - Inicializa su propio generador de números aleatorios.
     * - Crea un vector local para almacenar el punto generado.
     * - Construye distribuciones uniformes para cada dimensión.
     * - Evalúa una parte de las muestras asignadas mediante
     *   `#pragma omp for`.
     *
     * La cláusula `reduction(+:cont)` suma automáticamente las
     * contribuciones de todos los hilos al finalizar la región.
     */
#pragma omp parallel reduction(+:cont)
    {
        /// Generador de entropía.
        std::random_device rd;

        /// Generador Mersenne Twister independiente para cada hilo.
        std::mt19937 gen(rd() ^ omp_get_thread_num());

        /// Vector local que almacena un punto del espacio de integración.
        std::vector<double> comp(dimension);

        /// Distribuciones uniformes para cada dimensión.
        std::vector<std::uniform_real_distribution<double>> distribuciones;

        for (int j = 0; j < dimension; j++){
            distribuciones.push_back(
                std::uniform_real_distribution<double>(
                    vec[j].first,
                    vec[j].second
                )
            );
        }

        /**
         * @brief Distribución de las iteraciones entre los hilos.
         *
         * Cada hilo genera un subconjunto de las muestras aleatorias
         * y acumula localmente la suma de las evaluaciones.
         */
#pragma omp for
        for (int i = 0; i < N; i++){

            // Generación del punto aleatorio.
            for (int j = 0; j < dimension; j++){
                comp[j] = distribuciones[j](gen);
            }

            // Evaluación de la función.
            cont += func(comp);
        }
    }

    return g * cont;
}
