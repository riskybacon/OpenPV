#ifndef ACCUMULATOR_H_
#define ACCUMULATOR_H_

namespace PV {

template<typename T = float>
class AccumulatePre {
public:
   typedef T Type;
   void operator()(int kPreExt, int nk, T * v, T a, T * w, void * auxPtr, T dtFactor) {
      for (int k = 0; k < nk; k++) {
         v[k] += a * w[k];
      }
   }
};

template<typename T = float>
class AccumulatePost {
public:
   typedef T Type;
   void operator()(int kPreRes, int nk, T * v, T * a, T * w, void * auxPtr, T dtFactor) {
      T dv = 0;
      for (int k = 0; k < nk; k++) {
         dv += a[k] * w[k];
      }
      *v += dtFactor * dv;
   }
};

} // Namespace PV

#endif // ACCUMULATOR_H_
