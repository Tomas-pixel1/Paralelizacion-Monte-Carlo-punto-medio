#include <cstdlib>
#include <iostream>
#include <vector>


#include "MonteCarlo.hpp"

MonteCarlo::MonteCarlo(int d){
  dimension = d;
}

MonteCarlo::MonteCarlo(){
  std::cout<< "No es posible d = 0"<< std::endl;
}

MonteCarlo::MonteCarlo(const MonteCarlo &obj){
  std::cout << "constructor de copia invocado" << std::endl;
  int dim = obj.dimension;
}

MonteCarlo::~MonteCarlo(){
  std::cout << "Destructor invocado" << std::endl;
}

double MonteCarlo::Productorio(const std::vector<std::pair<double, double>>& vec){
  double volumen = 1.0;
  for (int i = 0; i < dimension; i++){
    volumen *= vec[i].second - vec[i].first;
  }
  return volumen;
}

double MonteCarlo::integral(const std::vector<std::pair<double, double>>& vec, int N, const std::function<double(std::vector<double>)>& func){
  double volumen = MonteCarlo::Productorio(vec);
  double g = volumen / N;
  double cont = 0.0;
  std::vector<std::vector<double>> x;
  
  for (int i = 0; i < N; i++){
    std::vector<double> comp;
    for (int j = 0; j < dimension; j++){
      comp.push_back(rand() * (vec[j].second - vec[j].first)/RAND_MAX + vec[j].first);
    }
    x.push_back(comp);
    cont += func(x[i]);
  }

  return g * cont;
}
