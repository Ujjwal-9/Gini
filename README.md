# Gini
Gini Implementation in C++, R, Python.

#### C++ Implementation:

```C++
double Gini(const std::vector &a, decltype(a) &p) {
  assert(a.size() == p.size());
  struct K {double a, p;} k[a.size()];
  for (auto i = 0; i != a.size(); ++i) k[i] = {a[i], p[i]};
  std::stable_sort(k, k+a.size(), [](const K &a, const K &b) {return a.p > b.p;});
  double accPopPercSum=0, accLossPercSum=0, giniSum=0, sum=0;
  for (auto &i: a) sum += i;
  for (auto &i: k) {
     accLossPercSum += i.a/sum;
     accPopPercSum += 1.0/a.size();
     giniSum += accLossPercSum-accPopPercSum;
  }
  return giniSum/a.size();
}
double GiniNormalized(const std::vector &a, decltype(a) &p) {
  return Gini(a, p)/Gini(a, a);
}
```

```C++
#include <vector>
#include <algorithm>
#include <assert.h>

double Gini(const std::vector<double> &a, decltype(a) &p) {
  assert(a.size() == p.size());

  // k[i] = { a[i],  p[i] };                                                                                                                   
  struct K {double a, p;} k[a.size()];
  for (auto i = 0; i != a.size(); ++i) k[i] = {a[i], p[i]};

  // sort(k) by descending p                                                                                                                   
  std::stable_sort(k, k+a.size(), [](const K &a, const K &b) {return a.p > b.p;});

  double accPopPercSum=0, accLossPercSum=0, giniSum=0, sum=0;

  // sum = accum(a);                                                                                                                           
  // or total actual sum                                                                                                                       
  for (auto &i: a) sum += i;

  // accLossPercSum = accum(i.a / sum);                                                                                                        
//   for (auto &i: k) {                                                                                                                        
  for (unsigned int fu = 0; fu < a.size(); ++fu) {
    struct K& i = k[fu];
    accLossPercSum += i.a/sum;
    accPopPercSum += 1.0/a.size();
    giniSum += accLossPercSum - accPopPercSum;
  }

  return giniSum/a.size();
}

double GiniNormalized(const std::vector<double> &a, decltype(a) &p) {
  return Gini(a, p)/Gini(a, a);
}

void GiniTest() {
  auto T = [](const std::vector<double> &a, decltype(a) p, double g, double n) {
    auto E = [](double a, double b, double e=1e-6) {return fabs(a-b) <= e;};
    assert(E(Gini(a, p), g) && E(GiniNormalized(a, p), n));
  };
  T({1, 2, 3}, {10, 20, 30}, 0.111111, 1);
  T({1, 2, 3}, {30, 20, 10}, -0.111111, -1);
  T({1, 2, 3}, {0, 0, 0}, -0.111111, -1);
  T({3, 2, 1}, {0, 0, 0}, 0.111111, 1);
  T({1, 2, 4, 3}, {0, 0, 0, 0}, -0.1, -0.8);
  T({2, 1, 4, 3}, {0, 0, 2, 1}, 0.125, 1);
  T({0, 20, 40, 0, 10}, {40, 40, 10, 5, 5}, 0, 0);
  T({40, 0, 20, 0, 10}, {1000000, 40, 40, 5, 5}, 0.171428, 0.6);
  T({40, 20, 10, 0, 0}, {40, 20, 10, 0, 0}, 0.285714, 1);
  T({1, 1, 0, 1}, {0.86, 0.26, 0.52, 0.32}, -0.041666, -0.333333);
}

int main(int ac, char** av)
{
  GiniTest();
}
```
compile with:
g++-4.6 -std=c++0x gini.cc

#### R Implementation:
```R
normalizedGini <- function(aa, pp) {
    Gini <- function(a, p) {
        if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
        temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
        temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
        population.delta <- 1 / length(a)
        total.losses <- sum(a)
        null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
        accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
        gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
        sum(gini.sum) / length(a)
    }
    Gini(aa,pp) / Gini(aa,aa)
}
```

#### Python Implementation:
```python
def gini(actual, pred, cmpcol = 0, sortcol = 1):
     assert( len(actual) == len(pred) )
     all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
     all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
     totalLosses = all[:,0].sum()
     giniSum = all[:,0].cumsum().sum() / totalLosses
 
     giniSum -= (len(actual) + 1) / 2.
     return giniSum / len(actual)
 
 def gini_normalized(a, p):
     return gini(a, p) / gini(a, a)
 
 def test_gini():
     def fequ(a,b):
         return abs( a -b) < 1e-6
     def T(a, p, g, n):
         assert( fequ(gini(a,p), g) )
         assert( fequ(gini_normalized(a,p), n) )
     T([1, 2, 3], [10, 20, 30], 0.111111, 1)
     T([1, 2, 3], [30, 20, 10], -0.111111, -1)
     T([1, 2, 3], [0, 0, 0], -0.111111, -1)
     T([3, 2, 1], [0, 0, 0], 0.111111, 1)
     T([1, 2, 4, 3], [0, 0, 0, 0], -0.1, -0.8)
     T([2, 1, 4, 3], [0, 0, 2, 1], 0.125, 1)
     T([0, 20, 40, 0, 10], [40, 40, 10, 5, 5], 0, 0)
     T([40, 0, 20, 0, 10], [1000000, 40, 40, 5, 5], 0.171428,
       0.6)
     T([40, 20, 10, 0, 0], [40, 20, 10, 0, 0], 0.285714, 1)
     T([1, 1, 0, 1], [0.86, 0.26, 0.52, 0.32], -0.041666,
       -0.333333)
```
