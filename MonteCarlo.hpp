#ifndef MONTECARLO_HPP
#define MONTECARLO_HPP
#include <vector>
#include <functional>

class MonteCarlo{
  public:
    MonteCarlo(int d);
    MonteCarlo(const MonteCarlo &obj);
    ~MonteCarlo();

    double integral(const std::vector<std::pair<double, double>>&, int N,const std::function<double(std::vector<double>)>& func);

  private:
    int dimension;
    MonteCarlo();
    double Productorio(const std::vector<std::pair<double, double>>&);
};

#endif

