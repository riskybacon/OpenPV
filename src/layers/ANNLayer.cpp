/*
 * ANNLayer.cpp
 *
 *  Created on: Dec 21, 2010
 *      Author: pschultz
 */

#include "ANNLayer.hpp"
#include "../layers/updateStateFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

void ANNLayer_update_state(
    const int numNeurons,
    const int nx,
    const int ny,
    const int nf,
    const int nb,

    float * V,
    const float Vth,
    const float VMax,
    const float VMin,
    const float VShift,
    float * GSynHead,
    float * activity);

#ifdef __cplusplus
}
#endif

namespace PV {

ANNLayer::ANNLayer() {
   initialize_base();
}

ANNLayer::ANNLayer(const char * name, HyPerCol * hc, int numChannels) {
   initialize_base();
   initialize(name, hc, numChannels);
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}  // end ANNLayer::ANNLayer(const char *, HyPerCol *)

ANNLayer::~ANNLayer() {}

int ANNLayer::initialize_base() {
   return PV_SUCCESS;
}

int ANNLayer::initialize(const char * name, HyPerCol * hc, int numChannels) {
   int status = HyPerLayer::initialize(name, hc, numChannels);
   assert(status == PV_SUCCESS);
   PVParams * params = parent->parameters();

   status |= readVThreshParams(params);
#ifdef PV_USE_OPENCL
   numEvents=NUM_ANN_EVENTS;
#endif
   return status;
}

#ifdef PV_USE_OPENCL
/**
 * Initialize OpenCL buffers.  This must be called after PVLayer data have
 * been allocated.
 */
int ANNLayer::initializeThreadBuffers(const char * kernel_name)
{
   int status = HyPerLayer::initializeThreadBuffers(kernel_name);

   //right now there are no ANN layer specific buffers...
   return status;
}

int ANNLayer::initializeThreadKernels(const char * kernel_name)
{
   char kernelPath[256];
   char kernelFlags[256];

   int status = CL_SUCCESS;
   CLDevice * device = parent->getCLDevice();

   const char * pvRelPath = "../PetaVision";
   sprintf(kernelPath, "%s/%s/src/kernels/%s.cl", parent->getPath(), pvRelPath, kernel_name);
   sprintf(kernelFlags, "-D PV_USE_OPENCL -cl-fast-relaxed-math -I %s/%s/src/kernels/", parent->getPath(), pvRelPath);

   // create kernels
   //
   krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
//kernel name should already be set correctly!
//   if (spikingFlag) {
//      krUpdate = device->createKernel(kernelPath, kernel_name, kernelFlags);
//   }
//   else {
//      krUpdate = device->createKernel(kernelPath, "Retina_nonspiking_update_state", kernelFlags);
//   }

   int argid = 0;

   status |= krUpdate->setKernelArg(argid++, getNumNeurons());
   status |= krUpdate->setKernelArg(argid++, clayer->loc.nx);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.ny);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.nf);
   status |= krUpdate->setKernelArg(argid++, clayer->loc.nb);


   status |= krUpdate->setKernelArg(argid++, clV);
   status |= krUpdate->setKernelArg(argid++, VThresh);
   status |= krUpdate->setKernelArg(argid++, VMax);
   status |= krUpdate->setKernelArg(argid++, VMin);
   status |= krUpdate->setKernelArg(argid++, VShift);
   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer());
//   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_EXC));
//   status |= krUpdate->setKernelArg(argid++, getChannelCLBuffer(CHANNEL_INH));
   status |= krUpdate->setKernelArg(argid++, clActivity);

   return status;
}
int ANNLayer::updateStateOpenCL(double time, double dt)
{
   int status = CL_SUCCESS;

   // wait for memory to be copied to device
   if (numWait > 0) {
       status |= clWaitForEvents(numWait, evList);
   }
   for (int i = 0; i < numWait; i++) {
      clReleaseEvent(evList[i]);
   }
   numWait = 0;

   status |= krUpdate->run(getNumNeurons(), nxl*nyl, 0, NULL, &evUpdate);
   krUpdate->finish();

   status |= getChannelCLBuffer()->copyFromDevice(1, &evUpdate, &evList[getEVGSyn()]);
   status |= clActivity->copyFromDevice(1, &evUpdate, &evList[getEVActivity()]);
   numWait += 2; //3;


   return status;
}
#endif

int ANNLayer::readVThreshParams(PVParams * params) {
   VMax = params->value(name, "VMax", max_pvdata_t);
   VThresh = params->value(name, "VThresh", -max_pvdata_t);
   if (VThresh > VMax) {
      VThresh = VMax;
      if (parent->columnId()==0) {
         fprintf(stderr, "Warning: ANNLayer \"%s\": VThresh > VMax.  VThresh changed to %f.\n", name, VMax);
      }
   }
   VMin = params->value(name, "VMin", VThresh);
   if (VMin > VThresh) {
      VMin = VThresh;
      if (parent->columnId()==0) {
         fprintf(stderr, "Warning: ANNLayer \"%s\": VMin > VThresh.  VMin changed to %f.\n", name, VThresh);
      }
   }
   VShift = params->value(name, "VShift", 0.0);
   if (VShift > VThresh-VMin) {
      VShift = VThresh-VMin;
      if (parent->columnId()==0) {
         fprintf(stderr, "Warning: ANNLayer \"%s\": VShift > VThresh-VMin.  VShift changed to %f.\n", name, VShift);
      }
   }
   return PV_SUCCESS;
}

int ANNLayer::updateState(double timed, double dt) {
    int status = PV_SUCCESS;
    // Check that there is at least two channels---exc and inh.
    // Can't put the check in allocateDataStructures because a subclass might need only one channel (e.g. HyPerLCA)
    // but subclasses still call parent class's allocateDataStructures
    // A subclass that allows only one channel should not call ANNLayer::updateState during its updateState call.
    if (getNumChannels()>=2) {
       status = HyPerLayer::updateState(timed, dt);
    }
    else {
       if (parent->columnId()==0) {
          fprintf(stderr, "ANNLayer \"%s\": At least two channels are needed but the layer has only %d.\n", name, getNumChannels());
       }
       status = PV_FAILURE;
    }
    MPI_Barrier(parent->icCommunicator()->communicator());
    if (status != PV_SUCCESS) exit(EXIT_FAILURE);
    return status;
}

//! new ANNLayer update state, to add support for GPU kernel.
//
/*!
 * REMARKS:
 *      - The kernel does the following:
//   HyPerLayer::updateV();
 *      - V = GSynExc - GSynInh
//   applyVMax(); (see below)
//   applyVThresh(); (see below)
 *      - Activity = V
 *      - GSynExc = GSynInh = 0
 *
 *
 */
int ANNLayer::doUpdateState(double time, double dt, const PVLayerLoc * loc, pvdata_t * A,
      pvdata_t * V, int num_channels, pvdata_t * gSynHead, bool spiking,
      unsigned int * active_indices, unsigned int * num_active)
{
   update_timer->start();
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag) {
      updateStateOpenCL(time, dt);
      //HyPerLayer::updateState(time, dt);
   }
   else {
#endif
      int nx = loc->nx;
      int ny = loc->ny;
      int nf = loc->nf;
      int num_neurons = nx*ny*nf;
      ANNLayer_update_state(num_neurons, nx, ny, nf, loc->nb, V, VThresh, VMax, VMin, VShift, gSynHead, A);
      if (this->writeSparseActivity){
         updateActiveIndices();  // added by GTK to allow for sparse output, can this be made an inline function???
      }
#ifdef PV_USE_OPENCL
   }
#endif

   update_timer->stop();
   return PV_SUCCESS;
}

int ANNLayer::setActivity() {
   const PVLayerLoc * loc = getLayerLoc();
   int nx = loc->nx;
   int ny = loc->ny;
   int nf = loc->nf;
   int nb = loc->nb;
   int num_neurons = nx*ny*nf;
   int status;
   status = setActivity_HyPerLayer(num_neurons, getCLayer()->activity->data, getV(), nx, ny, nf, nb);
   if( status == PV_SUCCESS ) status = applyVThresh_ANNLayer(num_neurons, getV(), VMin, VThresh, VShift, getCLayer()->activity->data, nx, ny, nf, nb);
   if( status == PV_SUCCESS ) status = applyVMax_ANNLayer(num_neurons, getV(), VMax, getCLayer()->activity->data, nx, ny, nf, nb);
   return status;
}


//int ANNLayer::updateV() {
//   HyPerLayer::updateV();
//   applyVMax();
//   applyVThresh();
//   return PV_SUCCESS;
//}

//int ANNLayer::applyVMax() {
//   if( VMax < FLT_MAX ) {
//      pvdata_t * V = getV();
//      for( int k=0; k<getNumNeurons(); k++ ) {
//         if(V[k] > VMax) V[k] = VMax;
//      }
//   }
//   return PV_SUCCESS;
//}

//int ANNLayer::applyVThresh() {
//   if( VThresh > -FLT_MIN ) {
//      pvdata_t * V = getV();
//      for( int k=0; k<getNumNeurons(); k++ ) {
//         if(V[k] < VThresh)
//            V[k] = VMin;
//      }
//   }
//   return PV_SUCCESS;
//}


}  // end namespace PV

///////////////////////////////////////////////////////
//
// implementation of ANNLayer kernels
//

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PV_USE_OPENCL
#  include "../kernels/ANNLayer_update_state.cl"
#else
#  undef PV_USE_OPENCL
#  include "../kernels/ANNLayer_update_state.cl"
#  define PV_USE_OPENCL
#endif

#ifdef __cplusplus
}
#endif

