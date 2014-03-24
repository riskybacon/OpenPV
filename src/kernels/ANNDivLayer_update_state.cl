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
// update the state of a retinal layer (spiking)
//
//    assume called with 1D kernel
//
CL_KERNEL
void ANNDivLayer_update_state(
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int nb,

    CL_MEM_GLOBAL float * V,
    const float Vth,
    const float AMax,
    const float AMin,
    CL_MEM_GLOBAL float * GSynHead,
//    CL_MEM_GLOBAL float * GSynExc,
//    CL_MEM_GLOBAL float * GSynInh,
//    CL_MEM_GLOBAL float * GSynInhB,
    CL_MEM_GLOBAL float * activity)
{


   //updateV():
   updateV_ANNDivInh(numNeurons, V, GSynHead);
   //setActivity():
   setActivity_HyPerLayer(numNeurons, activity, V, nx, ny, nf, nb);
   //resetGSynBuffers():
   resetGSynBuffers_HyPerLayer(numNeurons, 3, GSynHead);
}
