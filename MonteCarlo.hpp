#ifndef MONTECARLO_HPP
#define MONTECARLO_HPP
#include <vector>
#include <functional>

/**
 * @brief Clase para aproximar integrales multidimensionales usando el método de Monte Carlo.
 *
 * Implementa el método del valor medio de Monte Carlo, donde la integral se aproxima
 * mediante la evaluación de la función en N puntos aleatorios dentro del dominio de integración.
 */
class MonteCarlo{
  public:

    /**
     * @brief Constructor principal.
     * @param d Número de dimensiones del espacio de integración. Debe ser mayor a 0.
     */
    MonteCarlo(int d);

    /**
     * @brief Constructor de copia.
     * @param obj Objeto MonteCarlo a copiar.
     */
    MonteCarlo(const MonteCarlo &obj);

    /**
     * @brief Destructor.
     */
    ~MonteCarlo();

    /**
     * @brief Aproxima la integral de una función multidimensional usando Monte Carlo.
     *
     * Genera N puntos aleatorios uniformes dentro del hipercubo definido por los límites,
     * evalúa la función en cada punto y calcula el promedio ponderado por el volumen.
     *
     * @param vec Vector de pares (a, b) que define los límites de integración por dimensión.
     * @param N Número de puntos aleatorios a generar. A mayor N, menor error esperado (~1/√N).
     * @param func Función a integrar. Debe recibir un std::vector<double> y retornar un double.
     * @return Aproximación numérica de la integral.
     */
    double integral(const std::vector<std::pair<double, double>>& vec, int N,
                    const std::function<double(std::vector<double>)>& func);

  private:
    int dimension; ///< Número de dimensiones del espacio de integración.

    /**
     * @brief Constructor por defecto deshabilitado. No se permite crear un objeto sin dimensión.
     */
    MonteCarlo();

    /**
     * @brief Calcula el volumen del hipercubo de integración.
     *
     * Multiplica las longitudes de cada intervalo de integración: ∏(b_i - a_i).
     *
     * @param vec Vector de pares (a, b) con los límites por dimensión.
     * @return Volumen del dominio de integración.
     */
    double Productorio(const std::vector<std::pair<double, double>>& vec);
};

#endif
