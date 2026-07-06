/**
 * @file MonteCarloSM.cpp
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

#include "MonteCarlo.hpp"

/**
 * @ingroup MonteCarloOMP
 * @brief Constructor de la clase MonteCarlo.
 *
 * Inicializa la dimensión del espacio de integración.
 *
 * @param d Dimensión del dominio de integración.
 */
MonteCarlo::MonteCarlo(int d){
    dimension = d;
}

/**
 * @ingroup MonteCarloOMP
 * @brief Constructor por defecto.
 *
 * No permite crear objetos sin dimensión válida.
 */
MonteCarlo::MonteCarlo(){
    std::cout << "No es posible d = 0" << std::endl;
}

/**
 * @ingroup MonteCarloOMP
 * @brief Constructor de copia.
 *
 * Copia otro objeto MonteCarlo.
 */
MonteCarlo::MonteCarlo(const MonteCarlo &obj){
    std::cout << "constructor de copia invocado" << std::endl;
    int dim = obj.dimension;
}

/**
 * @ingroup MonteCarloOMP
 * @brief Destructor de la clase.
 */
MonteCarlo::~MonteCarlo(){
    std::cout << "Destructor invocado" << std::endl;
}

/**
 * @ingroup MonteCarloOMP
 * @brief Calcula el volumen de integración.
 */
double MonteCarlo::Productorio(const std::vector<std::pair<double, double>>& vec){
    double volumen = 1.0;

    for (int i = 0; i < dimension; i++){
        volumen *= vec[i].second - vec[i].first;
    }

    return volumen;
}

/**
 * @ingroup MonteCarloOMP
 * @brief Aproxima la integral mediante Monte Carlo con OpenMP.
 */
double MonteCarlo::integral(
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

/** @} */
