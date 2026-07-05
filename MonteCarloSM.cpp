#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>
#include <omp.h> //Se agrega la libreria de OpenMP
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
  
  // Iniciamos la región paralela 
  // Se agrega "reduction(+:cont)" para asegurar que cada hilo sume su propia copia de "cont" 
  // y al final se sumen de forma mas segura a la variable principal
  #pragma omp parallel reduction(+:cont)
  {
    // Iniciamos con la configuracion del generador aleatorio por hilo
    std::random_device rd;
    std::mt19937 gen(rd() ^ omp_get_thread_num());

    // Se crea un vector local para este hilo para vitar hacer push_back
    std::vector<double> comp(dimension);

    // Distribuciones uniformes para cada dimencion
    std::vector<std::uniform_real_distribution<double>> distribuciones;
    for (int j = 0; j < dimension; j++){
      distribuciones.push_back(std::uniform_real_distribution<double>(vec[j].first, vec[j].second));
    }

    // Se reparte las cargas del bucle entre los hilos 
    #pragma omp for 
    for (int i =0; i < N; i++){
      // Generar las coordenadas aleatorias
      for (int j = 0; j < dimension; j++){
      comp[j] = distribuciones[j](gen);
    }
    // Se evalua la funcion 
    cont += func(comp);
  }
  }
  
  return g * cont;
}
