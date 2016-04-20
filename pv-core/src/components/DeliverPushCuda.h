#ifndef DELIVER_PUSH_CUDA_H_
#define DELIVER_PUSH_CUDA_H_

#ifdef PV_USE_CUDA

#include <type_traits>
#include "connections/HyPerConn.hpp"
#include "layers/HyPerLayer.hpp"
#include "include/pv_common.h"
#include "include/pv_types.h"
#include "components/Deliver.h"
#include "components/ThreadBuffer.h"
#include "cudakernels/CudaRecvPost.hpp"
#include "cudakernels/CudaRecvPre.hpp"
#include "arch/cuda/CudaBuffer.hpp"

namespace PV {

/**
 * Push the pre-synaptic perspective over the synapse and accumulate onto the post-synaptic perspective.
 *
 * The push method of delivery has an optimization that allows for neurons in the pre-synaptice
 * layer to be skipped if they are not active. This can result in large performance gains
 * of DeliverPull, when this sort of sparsity is available.
 *
 * A downside to the push method is that multiple threads compete for writing output
 * to the same memory location.
 *
 * To get around this, each thread has its own buffer, and the buffers are summed up
 * into the post.
 *
 * ISSUES
 *
 * This class should not be dependent on HyPerConn. This creates a circular class
 * dependency and indicates that HyPerConn has grown too large.
 *
 * Dependencies
 *
 * - HyPerLayer: pre and post neurons)
 * - HyPerConn: weight values and data structure geometry, dtFactor
 * - PVLayerCube: pre and post neurons
 */
template<typename AccumulatorType, typename T = float>
class DeliverPushCuda : public Deliver {
   // Future implementation may allow for each of these types to be different.
   // These typedefs allow for a path forward with that implementation
   typedef T ActivityType;
   typedef T WeightType;
   typedef T PreType;
   typedef T PostType;

   // HyPerConn for whom propogation will be perfored
   HyPerConn *mConn;
   // Pre-synaptic layer
   HyPerLayer * mPre;
   // Post-synaptic layer
   HyPerLayer * mPost;
   // Number of threads that will be used
   int mNumThreads;
   // Number of batches
   int mNumBatches;
   // The accumulator function
   AccumulatorType mAccumulator;

public:
   DeliverPushCuda(HyPerConn * conn, HyPerLayer * pre, HyPerLayer * post, int numThreads, int numBatches)
   : mConn(conn)
   , mPre(pre)
   , mPost(post)
   , mNumThreads(numThreads)
   , mNumBatches(numBatches)
   {
   }

   void operator()(PVLayerCube const * activity, int arbor, int* numActive = NULL, int** activeList = NULL) {
      float dtFactor;
      if (mConn->getPvpatchAccumulateType() == ACCUMULATE_STOCHASTIC) {
         dtFactor = mConn->getParent()->getDeltaTime();
      }
      else if (mConn->getPvpatchAccumulateType() == ACCUMULATE_CONVOLVE) {
         dtFactor = mConn->getConvertToRateDeltaTimeFactor();
      }
      else{
         std::cout << "Pooling accumulate not implemented for GPUs";
         exit(-1);
      }

      PVCuda::CudaRecvPre *krRecvPre = mConn->getKrRecvPre();
      krRecvPre->set_dt_factor(dtFactor);

      // Post layer receives synaptic input
      // Only with respect to post layer
      const PVLayerLoc * preLoc = mPre->getLayerLoc();
      const PVLayerLoc * postLoc = mPost->getLayerLoc();
      // If the connection uses gpu to receive, update all buffers

      // TODO see if you can avoid this step of transferring patches to gpu
      // Based on arborId
      // Other way would be to just allocate all arbors to gpu

      // If more than 1 arbor, need to update patches and GSynPatchStart.
      // If one arbor, done in allocatePreKernel in HyPerConn
      if (mConn->numberOfAxonalArborLists() > 1) {
         PVPatch* h_patches = mConn->weights(arbor)[0]; //0 because it's one block of memory
         PVCuda::CudaBuffer * d_patches = mConn->getDevicePatches();
         pvAssert(d_patches);

         d_patches->copyToDevice(h_patches);

         size_t* h_GSynPatchStart = mConn->getGSynPatchStart()[arbor];
         PVCuda::CudaBuffer * d_GSynPatchStart = mConn->getDeviceGSynPatchStart();
         pvAssert(d_GSynPatchStart);
         d_GSynPatchStart->copyToDevice(h_GSynPatchStart);
      }

      //Update pre datastore, post gsyn, and conn weights
      //Only if their updated
      if (mPre->getUpdatedDeviceDatastoreFlag()) {
         float * h_preDatastore = activity->data;
         PVCuda::CudaBuffer * d_preDatastore = mPre->getDeviceDatastore();
         pvAssert(d_preDatastore);
         d_preDatastore->copyToDevice(h_preDatastore);

         //Copy active indices and num active if needed
         if (activity->isSparse) {
            PVCuda::CudaBuffer * d_ActiveIndices;
            PVCuda::CudaBuffer * d_numActive;
            d_ActiveIndices = mPre->getDeviceActiveIndices();
            d_numActive = mPre->getDeviceNumActive();
            pvAssert(d_ActiveIndices);
            unsigned int * h_ActiveIndices = activity->activeIndices;
            long * h_numActive = activity->numActive;
            pvAssert(h_ActiveIndices);
            d_numActive->copyToDevice(h_numActive);
            d_ActiveIndices->copyToDevice(h_ActiveIndices);
         }
         //Device now has updated
         mPre->setUpdatedDeviceDatastoreFlag(false);
      }

      //X direction is active neuron
      //Y direction is post patch size
      long totActiveNeuron[mConn->getParent()->getNBatch()];
      long maxTotalActiveNeuron = 0;
      for (int b = 0; b < mConn->getParent()->getNBatch(); b++) {
         if (activity->isSparse) {
            totActiveNeuron[b] = activity->numActive[b];
         } else {
            totActiveNeuron[b] = mPre->getNumExtended();
         }
         if(totActiveNeuron[b] > maxTotalActiveNeuron){
            maxTotalActiveNeuron = totActiveNeuron[b];
         }
      }
      
      long totPatchSize = mConn->xPatchSize() * mConn->yPatchSize() * mConn->fPatchSize();
      
      long totThreads = maxTotalActiveNeuron * totPatchSize;
      
      //krRecvPre->set_numActive(totActiveNeuron);
      
      int maxThreads = mConn->getParent()->getDevice()->get_max_threads();
      int numLocalThreads = totPatchSize < maxThreads ? totPatchSize : maxThreads;
      
      krRecvPre->run_nocheck(totThreads, numLocalThreads);
   }
};
} // Namespace PV

#endif // PV_USE_CUDA
#endif // _DELIVER_PUSH_CUDA_H
