#include <iostream>
#include <mpi.h>
#include "MonteCarlo.hpp"

double fx(std::vector<double> vec){
  return vec[0] * vec[0] + vec[1] * vec[1];
}
int main(int argc, char*argv[]){
 MPI_Init(&argc, &argv);
 int rank,size;
 double time1=MPI_Wtime();
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 MPI_Comm_size(MPI_COMM_WORLD, &size);
 MonteCarlo dimensiones =  MonteCarlo(2);
 std::vector<std::pair<double, double>> limites = {{0.0, 1.0}, {0.0, 1.0}};
 if (rank ==0)
   std::cout << "Dimensiones: 2" << std::endl;
  
for (int N = 1; N<1000000000; N*=10){
   double I = dimensiones.integral(limites, N, fx);
   double time2 = MPI_Wtime();
   if (rank == 0){
      std::cout<<"Para "<<N<<" iteraciones -> "<<I<<std::endl;
      std::cout<<"Tiempo --> "<<time2 - time1<<std::endl;
}	
}

  MPI_Finalize();
  return 0;
}
