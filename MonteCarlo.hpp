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
