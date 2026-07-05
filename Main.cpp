#include <iostream>

#include "MonteCarlo.hpp"

double fx(std::vector<double> vec){
  return vec[0] * vec[0] + vec[1] * vec[1];
}
int main(){
 MonteCarlo dimensiones =  MonteCarlo(2);
 std::vector<std::pair<double, double>> limites = {{0.0, 1.0}, {0.0, 1.0}};
 std::cout << "Dimensiones: 2" << std::endl;

  
for (int N = 1; N<1000000000; N*=10){
   double I = dimensiones.integral(limites, N, fx);
   std::cout << I << std::endl;
}
  
  return 0;
}
