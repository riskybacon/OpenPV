#include "../layers/updateStateFunctions.h"


#ifndef PV_USE_OPENCL
#  define CL_KERNEL
#  define CL_MEM_GLOBAL
#  define CL_MEM_CONST
#  define CL_MEM_LOCAL
#else  /* compiling with OpenCL */
#  define CL_KERNEL       __kernel
#  define CL_MEM_GLOBAL   __global
#  define CL_MEM_CONST    __constant
#  define CL_MEM_LOCAL    __local
//#  include "conversions.hcl"
#endif


//
// update the state of an ANNWhitened layer
//
// To allow porting to GPUs, functions called from this function must be
// static inline functions.  If a subclass needs new behavior, it needs to
// have its own static inline function.
//
CL_KERNEL
void ANNWhitenedLayer_update_state(
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int nb,

    CL_MEM_GLOBAL float * V,
    const float Vth,
    const float AMax,
    const float AMin,
    const float AShift,
    const float VWidth,
    CL_MEM_GLOBAL float * GSynHead,
    CL_MEM_GLOBAL float * activity)
{
   updateV_ANNWhitenedLayer(numNeurons, V, GSynHead, activity, AMax, AMin, Vth, AShift, VWidth, nx, ny, nf, nb);
}
