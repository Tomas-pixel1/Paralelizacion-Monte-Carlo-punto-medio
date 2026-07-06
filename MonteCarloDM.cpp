#include <cstdlib>
#include <iostream>
#include <vector>
#include <mpi.h>


#include "MonteCarlo.hpp"

MonteCarlo::MonteCarlo(int d){
  dimension = d;
}

MonteCarlo::MonteCarlo(){
  std::cout<< "No es posible d = 0"<< std::endl;
}

MonteCarlo::MonteCarlo(const MonteCarlo &obj){
  std::cout << "constructor de copia invocado" << std::endl;
  dimension = obj.dimension;
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
  int size, rank;
  //Paralelización aquí debido al alto número de iteraciones realizadas
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  int Nlocal = N/size;
  int rest = N%size;
  
  if (rest && rank < rest)
    Nlocal ++;

  int start = Nlocal * rank;
 
  if (rest && rank >= rest)
    start += rest;
 
  int end = start + Nlocal;
  
  double volumen = MonteCarlo::Productorio(vec);
  double contLocal = 0.0;
  std::vector<double> comp(vec.size());//el tamaño del vector indica la dimensión
  for (int i = start; i < end; i++){
    for (int j = 0; j < dimension; j++){
      comp[j]=rand() * (vec[j].second - vec[j].first)/RAND_MAX + vec[j].first;
    }	
    contLocal += func(comp);//Evaluar la función en el número aleatorio y sumarlo a la acumulación local
  
  }
  double cont=0.0;
  MPI_Reduce(&contLocal,&cont,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);//Sumar cada acumulación de cada núcleo en la variable cont
  if (rank == 0)
    return volumen * cont / N;
  return 0;
}
