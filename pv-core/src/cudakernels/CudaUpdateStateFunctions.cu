#include "CudaUpdateStateFunctions.hpp"
#include "../arch/cuda/cuda_util.hpp"
#include "conversions.hcu"



namespace PVCuda{
//Include update state functions with cuda flag on 
#include "../layers/updateStateFunctions.h"

//The actual wrapper kernel code thats calling updatestatefunctions
__global__
void HyPerLCALayer_update_state(
				const int nbatch,
				const int numNeurons,
				const int nx,
				const int ny,
				const int nf,
				const int lt,
				const int rt,
				const int dn,
				const int up,
				const int numChannels,
				float * V,
				const int numVertices,
				float * verticesV,
				float * verticesA,
				float * slopes,
				const bool selfInteract,
				double * dtAdapt,
				const float tau,
				float * GSynHead,
				float * activity)
{

   if((blockIdx.x * blockDim.x) + threadIdx.x < numNeurons*nbatch){
      updateV_HyPerLCALayer(
			    nbatch,
			    numNeurons,
			    numChannels,
			    V,
			    GSynHead,
			    activity,
			    numVertices,
			    verticesV,
			    verticesA,
			    slopes,
			    dtAdapt,
			    tau,
			    selfInteract,
			    nx,
			    ny,
			    nf,
			    lt,
			    rt,
			    dn,
			    up);
   }
}

__global__
void MomentumLCALayer_update_state(
				const int nbatch,
				const int numNeurons,
				const int nx,
				const int ny,
				const int nf,
				const int lt,
				const int rt,
				const int dn,
				const int up,
				const int numChannels,
				float * V,
            float * prevDrive,
				const int numVertices,
				float * verticesV,
				float * verticesA,
				float * slopes,
				const bool selfInteract,
				double * dtAdapt,
				const float tau,
            const float LCAMomentumRate,
				float * GSynHead,
				float * activity)
{

   if((blockIdx.x * blockDim.x) + threadIdx.x < numNeurons*nbatch){
      updateV_MomentumLCALayer(
			    nbatch,
			    numNeurons,
			    numChannels,
			    V,
			    GSynHead,
			    activity,
             prevDrive,
			    numVertices,
			    verticesV,
			    verticesA,
			    slopes,
			    dtAdapt,
			    tau,
             LCAMomentumRate,
			    selfInteract,
			    nx,
			    ny,
			    nf,
			    lt,
			    rt,
			    dn,
			    up);
   }
}



__global__
void ISTALayer_update_state(
			    const int nbatch,
			    const int numNeurons,
			    const int nx,
			    const int ny,
			    const int nf,
			    const int lt,
			    const int rt,
			    const int dn,
			    const int up,
			    const int numChannels,
			    float * V,
			    const float Vth,
			    double * dtAdapt,
			    const float tau,
			    float * GSynHead,
			    float * activity)
  {
    if((blockIdx.x * blockDim.x) + threadIdx.x < numNeurons*nbatch){
      updateV_ISTALayer(nbatch, 
			numNeurons, 
			V, 
			GSynHead, 
			activity,
			Vth, 
			dtAdapt, 
			tau, 
			nx, 
			ny, 
			nf, 
			lt, 
			rt, 
			dn, 
			up, 
			numChannels);
    }
  }

CudaUpdateHyPerLCALayer::CudaUpdateHyPerLCALayer(CudaDevice* inDevice):CudaKernel(inDevice){
}
  
CudaUpdateHyPerLCALayer::~CudaUpdateHyPerLCALayer(){
}

CudaUpdateMomentumLCALayer::CudaUpdateMomentumLCALayer(CudaDevice* inDevice):CudaKernel(inDevice){
}
  
CudaUpdateMomentumLCALayer::~CudaUpdateMomentumLCALayer(){
}

CudaUpdateISTALayer::CudaUpdateISTALayer(CudaDevice* inDevice):CudaKernel(inDevice){
}

CudaUpdateISTALayer::~CudaUpdateISTALayer(){
}

void CudaUpdateHyPerLCALayer::setArgs(
				      const int nbatch,
				      const int numNeurons,
				      const int nx,
				      const int ny,
				      const int nf,
				      const int lt,
				      const int rt,
				      const int dn,
				      const int up,
				      const int numChannels,
				      
				      /* float* */ CudaBuffer* V,
				      
				      const int numVertices,
				      /* float* */ CudaBuffer* verticesV,
				      /* float* */ CudaBuffer* verticesA,
				      /* float* */ CudaBuffer* slopes,
				      const bool selfInteract,
				      /* double* */ CudaBuffer* dtAdapt,
				      const float tau,
				      
				      /* float* */ CudaBuffer* GSynHead,
				      /* float* */ CudaBuffer* activity
				      ){
  params.nbatch = nbatch;
  params.numNeurons = numNeurons;
  params.nx = nx;
  params.ny = ny;
  params.nf = nf;
  params.lt = lt;
  params.rt = rt;
  params.dn = dn;
  params.up = up;
  params.numChannels = numChannels;
  
  params.V = (float*) V->getPointer();
  
   params.numVertices = numVertices;
   params.verticesV = (float*) verticesV->getPointer();
   params.verticesA = (float*) verticesA->getPointer();
   params.slopes = (float*) slopes->getPointer();
   params.selfInteract = selfInteract;
   params.dtAdapt = (double*) dtAdapt->getPointer();
   params.tau = tau;
   
   params.GSynHead = (float*) GSynHead->getPointer();
   params.activity = (float*) activity->getPointer();
   
   setArgsFlag();
}


int CudaUpdateHyPerLCALayer::do_run(){
   int currBlockSize = device->get_max_threads();
   //Ceil to get all weights
   int currGridSize = ceil(((float)params.numNeurons * params.nbatch)/currBlockSize);
   //Call function
   HyPerLCALayer_update_state<<<currGridSize, currBlockSize, 0, device->getStream()>>>(
   params.nbatch,
   params.numNeurons,
   params.nx,
   params.ny,
   params.nf,
   params.lt,
   params.rt,
   params.dn,
   params.up,
   params.numChannels,
   params.V,
   params.numVertices,
   params.verticesV,
   params.verticesA,
   params.slopes,
   params.selfInteract,
   params.dtAdapt,
   params.tau,
   params.GSynHead,
   params.activity);
   handleCallError("HyPerLCALayer Update run");
   return 0;
}

void CudaUpdateMomentumLCALayer::setArgs(
				      const int nbatch,
				      const int numNeurons,
				      const int nx,
				      const int ny,
				      const int nf,
				      const int lt,
				      const int rt,
				      const int dn,
				      const int up,
				      const int numChannels,
				      
				      /* float* */ CudaBuffer* V,
                  /* float* */ CudaBuffer* prevDrive,
				      
				      const int numVertices,
				      /* float* */ CudaBuffer* verticesV,
				      /* float* */ CudaBuffer* verticesA,
				      /* float* */ CudaBuffer* slopes,
				      const bool selfInteract,
				      /* double* */ CudaBuffer* dtAdapt,
				      const float tau,
                  const float LCAMomentumRate,
				      
				      /* float* */ CudaBuffer* GSynHead,
				      /* float* */ CudaBuffer* activity
				      ){
  params.nbatch = nbatch;
  params.numNeurons = numNeurons;
  params.nx = nx;
  params.ny = ny;
  params.nf = nf;
  params.lt = lt;
  params.rt = rt;
  params.dn = dn;
  params.up = up;
  params.numChannels = numChannels;
  
  params.V = (float*) V->getPointer();
  params.prevDrive = (float*) prevDrive->getPointer();
  
   params.numVertices = numVertices;
   params.verticesV = (float*) verticesV->getPointer();
   params.verticesA = (float*) verticesA->getPointer();
   params.slopes = (float*) slopes->getPointer();
   params.selfInteract = selfInteract;
   params.dtAdapt = (double*) dtAdapt->getPointer();
   params.tau = tau;
   params.LCAMomentumRate = LCAMomentumRate;
   
   params.GSynHead = (float*) GSynHead->getPointer();
   params.activity = (float*) activity->getPointer();
   
   setArgsFlag();
}


int CudaUpdateMomentumLCALayer::do_run(){
   int currBlockSize = device->get_max_threads();
   //Ceil to get all weights
   int currGridSize = ceil(((float)params.numNeurons * params.nbatch)/currBlockSize);
   //Call function
   MomentumLCALayer_update_state<<<currGridSize, currBlockSize, 0, device->getStream()>>>(
   params.nbatch,
   params.numNeurons,
   params.nx,
   params.ny,
   params.nf,
   params.lt,
   params.rt,
   params.dn,
   params.up,
   params.numChannels,
   params.V,
   params.prevDrive,
   params.numVertices,
   params.verticesV,
   params.verticesA,
   params.slopes,
   params.selfInteract,
   params.dtAdapt,
   params.tau,
   params.LCAMomentumRate,
   params.GSynHead,
   params.activity);
   handleCallError("HyPerLCALayer Update run");
   return 0;
}

void CudaUpdateISTALayer::setArgs(
				  const int nbatch,
				  const int numNeurons,
				  const int nx,
				  const int ny,
				  const int nf,
				  const int lt,
				  const int rt,
				  const int dn,
				  const int up,
				  const int numChannels,
				  
				  /* float* */ CudaBuffer* V,
				  
				  const float Vth,
				  /* double* */ CudaBuffer* dtAdapt,
				  const float tau,
				  
				  /* float* */ CudaBuffer* GSynHead,
				  /* float* */ CudaBuffer* activity
				  ){
  params.nbatch = nbatch;
  params.numNeurons = numNeurons;
  params.nx = nx;
  params.ny = ny;
  params.nf = nf;
  params.lt = lt;
  params.rt = rt;
  params.dn = dn;
  params.up = up;
  params.numChannels = numChannels;
  
  params.V = (float*) V->getPointer();
  
  params.Vth = Vth;
  params.dtAdapt = (double*) dtAdapt->getPointer();
  params.tau = tau;
  
  params.GSynHead = (float*) GSynHead->getPointer();
  params.activity = (float*) activity->getPointer();
    
  setArgsFlag();
}

int CudaUpdateISTALayer::do_run(){
    int currBlockSize = device->get_max_threads();
    //Ceil to get all weights                                                                                           
    int currGridSize = ceil(((float)params.numNeurons * params.nbatch)/currBlockSize);
    //Call function
    ISTALayer_update_state<<<currGridSize, currBlockSize, 0, device->getStream()>>>(
    params.nbatch,
    params.numNeurons,
    params.nx,
    params.ny,
    params.nf,
    params.lt,
    params.rt,
    params.dn,
    params.up,
    params.numChannels,
    params.V,
    params.Vth,
    params.dtAdapt,
    params.tau,
    params.GSynHead,
    params.activity);
    handleCallError("ISTALayer Update run");
    return 0;
}

}
