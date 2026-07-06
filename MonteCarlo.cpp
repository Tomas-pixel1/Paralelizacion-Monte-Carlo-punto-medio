/**
 * @page Monte Carlo secuencial
 *
 * @brief Implementación del método de integración de Monte Carlo de punto medio en versión secuencial.
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
 * - Genera `N` muestras aleatorias uniformemente distribuidas.
 * - Evalúa la función de usuario en cada muestra.
 * - Obtiene el promedio de las evaluaciones y lo multiplica por el
 *   volumen de integración.
 *
 * ## Características
 * - Implementación completamente secuencial.
 * - Utiliza `rand()` para generar números pseudoaleatorios.
 * - No emplea paralelización ni sincronización entre hilos.
 *
 * ## Complejidad
 * - Tiempo: O(N · d)
 * - Espacio: O(N · d), debido al almacenamiento temporal de las muestras.
 *
 * @date 2026
 */

#include <cstdlib>
#include <iostream>
#include <vector>

#include "MonteCarlo.hpp"

/**
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
 * @brief Destructor de la clase.
 *
 * Libera los recursos asociados al objeto e informa de la destrucción
 * del mismo.
 */
MonteCarlo::~MonteCarlo(){
  std::cout << "Destructor invocado" << std::endl;
}

/**
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
double MonteCarlo::Productorio(const std::vector<std::pair<double, double>>& vec){
  double volumen = 1.0;

  for (int i = 0; i < dimension; i++){
    volumen *= vec[i].second - vec[i].first;
  }

  return volumen;
}

/**
 * @brief Aproxima una integral múltiple mediante el método de Monte Carlo.
 *
 * El algoritmo genera @p N puntos aleatorios distribuidos uniformemente
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
