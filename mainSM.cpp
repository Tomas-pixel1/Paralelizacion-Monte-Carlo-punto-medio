#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>

#include "MonteCarlo.hpp"

double fx(std::vector<double> vec){
  return vec[0] * vec[0] + vec[1] * vec[1];
}
int main(){
 MonteCarlo dimensiones =  MonteCarlo(2);
 std::vector<std::pair<double, double>> limites = {{0.0, 1.0}, {0.0, 1.0}};
 
 // Medir el impacto de la paralelización 
 int N = 100000000;
 int max_hilos = omp_get_max_threads();

 std::ofstream archivo("tiempos.txt");
  
  std::cout << "Iniciando escalabilidad con N = " << N << std::endl;
  std::cout << "Hilos disponibles: " << max_hilos << std::endl;
  
  for (int hilos = 1; hilos <= max_hilos; hilos++){
    omp_set_num_threads(hilos);
    
    // Medición de tiempo 
    double inicio = omp_get_wtime();
    double I = dimensiones.integral(limites, N, fx);
    double fin = omp_get_wtime();
    
    double tiempo_ejecucion = fin - inicio;
    
    std::cout << "Hilos: " << hilos << " | Tiempo: " << tiempo_ejecucion << "s | Integral: " << I << std::endl;
    
    archivo << hilos << " " << tiempo_ejecucion << "\n";
  }
  
  archivo.close();
  return 0;
}
