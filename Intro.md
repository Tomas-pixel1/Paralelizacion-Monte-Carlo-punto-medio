# Paralelizacion-Monte-Carlo-punto-medio

**Integrantes:**

Samuel Díaz Estrada - C12536
Sebastián Elizondo Peralta - C4E856
Sebastián Schult Revillat - C4J897
Tomás Salgado Gutiérrez - C27169


## Objetivos

1. Emplear el método de Monte Carlo de punto medio para evaluar una integral multi dimesional.  
2. Estudiar como se comporta el error en funcion del número de puntos $N$.  
3. Acelerar el proceso por medio del paralelismo de memoria compartida y evaluar su escalabilidad.
4. Acelerar el proceso por medio del paralelismo de memoria
 distribuida y evaluar su escalabilidad.

## Método Monte Carlo de punto medio

El método de Monte Carlo de punto medio busca la aproximación de una integral por medio del cálculo del promedio o punto medio de la función, dado por la siguiente ecuación:

$$\bar{f} = \frac1{b-a}\int_a^b f(x) \text d x $$

Entonces para calcular el valor de la integral de forma discreta, se tiene que: 

$$\bar{f}\approx\frac1N\sum_{i=1}^N f(x_i) \ \Rightarrow \ I\approx \frac{b-a}{N}\sum_{i=1}^N f(x_i).$$

Donde los puntos $$x_i$$ son puntos pseudoaleatorios.
