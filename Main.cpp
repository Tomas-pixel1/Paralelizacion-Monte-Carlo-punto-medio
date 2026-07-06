#include <iostream>

#include "MonteCarlo.hpp"

/**
 * @ingroup MonteCarloSec
 * @brief Función a integrar.
 *
 * Calcula la suma de los cuadrados de las dos componentes del vector
 * de entrada:
 *
 * \f[
 * f(x,y)=x^2+y^2
 * \f]
 *
 * Esta función se utiliza como ejemplo para probar la implementación
 * del método de integración de Monte Carlo.
 *
 * @param vec Vector de dos componentes que representa el punto
 * \f$(x,y)\f$ donde se evalúa la función.
 *
 * @return Valor de \f$x^2+y^2\f$.
 */
double fx(std::vector<double> vec){
    return vec[0] * vec[0] + vec[1] * vec[1];
}

/**
 * @brief Función principal del programa.
 *
 * Crea un objeto de la clase MonteCarlo para realizar una integración
 * bidimensional de la función
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
 * Posteriormente ejecuta el algoritmo de Monte Carlo utilizando un
 * número creciente de muestras (potencias de diez) con el objetivo de
 * observar la convergencia de la aproximación numérica.
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
     * Cada par representa los límites inferior y superior de una
     * dimensión del dominio.
     *
     * En este caso:
     * - Primera dimensión: [0,1]
     * - Segunda dimensión: [0,1]
     */
    std::vector<std::pair<double, double>> limites = {
        {0.0, 1.0},
        {0.0, 1.0}
    };

    std::cout << "Dimensiones: 2" << std::endl;

    /**
     * @brief Bucle de experimentación.
     *
     * Ejecuta el algoritmo incrementando el número de muestras
     * aleatorias en un factor de diez en cada iteración:
     *
     * 1, 10, 100, ..., 10⁹.
     *
     * Para cada valor de N se calcula e imprime la aproximación de la
     * integral.
     */
    for (int N = 1; N < 1000000000; N *= 10){

        /// Aproximación de la integral mediante Monte Carlo.
        double I = dimensiones.integral(limites, N, fx);

        /// Impresión del resultado obtenido.
        std::cout << I << std::endl;
    }

    return 0;
}
