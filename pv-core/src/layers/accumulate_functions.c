#include "accumulate_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef COMPRESS_PHI
void pvpatch_accumulate(int kPreExt, int nk, float* restrict v, float a, pvwdata_t* restrict w,
                        float* restrict m)
{
   const float scale = 33.3;
   const float inv_scale = 1.0/scale;
   const float shift = 2.0;
   int k;

   for (k = 0; k < nk; k++) {
            v[k] = (((shift + scale*v[k]) + a*w[k]*m[k])
                  - shift) * inv_scale;
      // without mask
      //      v[k] = (((shift + scale*v[k]) + a*w[k])
      //                  - shift) * inv_scale;
   }
}
#else
  int pvpatch_accumulate(int kPreExt, int nk, float* RESTRICT v, float a, pvwdata_t* RESTRICT w, void * auxPtr, int sf)
{
   int k;
   int err = 0;
   float accumval = 0;
   for (k = 0; k < nk; k++) {
      accumval = a*w[k];
      v[k] += accumval;
   }
   return err;
}
#endif

int pvpatch_accumulate_from_post(int kPreExt, int nk, float * RESTRICT v, float * RESTRICT a, pvwdata_t * RESTRICT w, float dt_factor, void * auxPtr, int sf) {
   int status = 0;
   int k;
   float dv = 0.0;
   for (k = 0; k < nk; k++) {
      dv += a[k] * w[k];
   }
   *v += dt_factor * dv;
   return status;
}

int pvpatch_accumulate2(int nk, float* RESTRICT v, float a, pvwdata_t* RESTRICT w, float* RESTRICT m)
{
   int k;
   int err = 0;
   float accumval = 0;
   for (k = 0; k < nk; k++) {
      accumval = a*w[k]*m[k];
//#ifdef PV_USE_OPENMP_THREADS
//#pragma omp atomic
//#endif
      v[k] += accumval;
   }
      //v[k] += accumval;
   return err;
}

int pvpatch_accumulate_stochastic(int kPreExt, int nk, float* RESTRICT v, float a, pvwdata_t* RESTRICT w, void * auxPtr, int sf)
{
   taus_uint4 * rng = (taus_uint4 *) auxPtr;
   long along = (long) (a*cl_random_max());
   int err = 0;
   int k;
   float accumval = 0;
   for (k = 0; k < nk; k+=sf) {
      *rng = cl_random_get(*rng);
      accumval = (rng->s0 < along)*w[k];
//#ifdef PV_USE_OPENMP_THREADS
//#pragma omp atomic
//#endif
      v[k] += accumval;
   }
   return err;
}

int pvpatch_accumulate_stochastic_from_post(int kPreExt, int nk, float * RESTRICT v, float * RESTRICT a, pvwdata_t * RESTRICT w, float dt_factor, void * auxPtr, int sf) {
   int status = 0;
   taus_uint4 * rng = (taus_uint4 *) auxPtr;
   int k;
   float dv = 0.0f;
   for (k = 0; k < nk; k+=sf) {
      *rng = cl_random_get(*rng);
      double p = (double) rng->s0/cl_random_max(); // 0.0 < p < 1.0
      dv += (p<a[k]*dt_factor)*w[k];
   }
   *v = *v + dv;
   return status;
}

int pvpatch_max_pooling(int kPreGlobalExt, int nk, float* RESTRICT v, float a, pvwdata_t* RESTRICT w, void * auxPtr, int sf)
{
  int k;
  int err = 0;
  //float * gate = (float *)auxPtr;
  int * gate = (int*)auxPtr;
  for (k = 0; k < nk; k+=sf) {
    //     v[k] = v[k] > a*w[k] ? v[k] : a*w[k];
    //v[k] = v[k] > a*w[0] ? v[k] : a*w[0];
    float checkVal = a*w[0];
    if(v[k] <= checkVal){
       v[k] = checkVal;
       if(gate){
          gate[k] = kPreGlobalExt;
          //printf("Writing to address %p idx %d, writing to address %p val%f\n", &(gate[k]), kPreGlobalExt, &(v[k]), checkVal);
       }
    }
  }
  return err;
}

int pvpatch_max_pooling_from_post(int kPreGlobalExt, int nk, float * RESTRICT v, float * RESTRICT a, pvwdata_t * RESTRICT w, float dt_factor, void * auxPtr, int sf) {
   int status = 0;
   int k;
   float vmax = *v;
   int* gate = (int*)auxPtr;
   int gateMax;
   if(gate){
      gateMax = *gate;
   }
   for (k = 0; k < nk; k+=sf) {
      //vmax = vmax > a[k]*w[k] ? vmax : a[k]*w[k];
      //vmax = vmax > a[k]*w[0] ? vmax : a[k]*w[0];
      float checkVal = a[k]*w[0];
      if(vmax <= checkVal){
         vmax = checkVal;
         if(gate){
            gateMax = kPreGlobalExt+k;
         }
      }
   }
   *v = vmax;
   if(gate){
      *gate = gateMax;
   }
   return status;
}

int pvpatch_sum_pooling(int kPreExt, int nk, float* RESTRICT v, float a, pvwdata_t* RESTRICT w, void * auxPtr, int sf)
{
   int k;
   int err = 0;
   float accumval = 0;
   for (k = 0; k < nk; k+=sf) {
      accumval = a*w[0];
      v[k] += accumval;
   }
   return err;
}

int pvpatch_sumpooling_from_post(int kPreExt, int nk, float * RESTRICT v, float * RESTRICT a, pvwdata_t * RESTRICT w, float dt_factor, void * auxPtr, int sf) {
   int status = 0;
   int k;
   float dv = 0.0f;
   for (k = 0; k < nk; k+=sf) {
      dv += a[k]*w[0];
   }
   *v += dt_factor*dv;
   return status;
}



#ifdef __cplusplus
}
#endif
