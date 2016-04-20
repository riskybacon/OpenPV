#ifndef DELIVER_PULL_CUDA_H_
#define DELIVER_PULL_CUDA_H_

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
 * Pull the pre-synaptic perspective over the synapse and accumulate onto the post-synaptic perspective.
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
template<typename T = float>
class DeliverPullCuda : public Deliver {
   // Future implementation may allow for each of these types to be different.
   // These typedefs allow for a path forward with that implementation
   typedef T ActivityType;
   typedef T WeightType;
   typedef T PreType;
   typedef T PostType;

   // HyPerConn for whom propogation will be perfored
   HyPerConn *mConn;
   // The HyPerConn's post connection
   HyPerConn *mPostConn;
   // Pre-synaptic layer
   HyPerLayer * mPre;
   // Post-synaptic layer
   HyPerLayer * mPost;
   // Number of threads that will be used
   int mNumThreads;
   // Number of batches
   int mNumBatches;

public:
   DeliverPullCuda(HyPerConn * conn, HyPerLayer * pre, HyPerLayer * post, int numThreads, int numBatches)
   : mConn(conn)
   , mPostConn(conn->getPostConn())
   , mPre(pre)
   , mPost(post)
   , mNumThreads(numThreads)
   , mNumBatches(numBatches)
   {
   }

   void operator()(PVLayerCube const * activity, int arbor, int* numActive = NULL, int** activeList = NULL) {
      // Get number of neurons restricted target
      const int numRestricted = mPost->getNumNeurons();

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

      PVCuda::CudaRecvPost *krRecvPost = mConn->getKrRecvPost();
      pvAssert(krRecvPost);
      krRecvPost->set_dt_factor(dtFactor);

      const PVLayerLoc * sourceLoc = mPre->getLayerLoc();
      const PVLayerLoc * targetLoc = mPost->getLayerLoc();
      const PVHalo * sourceHalo = &sourceLoc->halo;

      const int sourceNx = sourceLoc->nx;
      const int sourceNy = sourceLoc->ny;
      const int sourceNf = sourceLoc->nf;
      const int targetNx = targetLoc->nx;
      const int targetNy = targetLoc->ny;
      const int targetNf = targetLoc->nf;

      // Get source layer's extended y stride
      int sy = (sourceNx + sourceHalo->rt + sourceHalo->lt) * sourceNf;
      // Get source layer's patch y stride
      int syp = mPostConn->yPatchStride();
      // Iterate through y patch
      int numPerStride = mPostConn->xPatchSize() * mPostConn->fPatchSize();

      long * startSourceExtBuf = mConn->getPostToPreActivity();
      if (!startSourceExtBuf) {
         std::cout << "HyPerLayer::recvFromPost error getting preToPostActivity from connection. Is shrink_patches on?\n";
         exit(EXIT_FAILURE);
      }

      bool updatePreAct = false;
      // Update pre activity, post gsyn, and conn weights
      // Only if they're updated
      if (mPre->getUpdatedDeviceDatastoreFlag()) {
         float * h_preDatastore = activity->data;
         PVCuda::CudaBuffer* d_preDatastore = mPre->getDeviceDatastore();
         pvAssert(d_preDatastore);
         d_preDatastore->copyToDevice(h_preDatastore);
         // Device now has updated
         mPre->setUpdatedDeviceDatastoreFlag(false);
         updatePreAct = true;
      }

#ifdef PV_USE_CUDNN
      // Permutation buffer is local to the kernel, NOT the layer
      // Therefore, we must permute Datastore every time
      krRecvPost->permuteDatastorePVToCudnn();
      //}

      // Permute GSyn
      krRecvPost->permuteGSynPVToCudnn(mConn->getChannel());
#endif // PV_USE_CUDNN

      int totF = targetNf;
      int totX = targetNx;
      int totY = targetNy;
      //Make sure local sizes are divisible by f, x, and y
      krRecvPost->run(totX, totY, totF, mConn->getNumXLocal(), mConn->getNumYLocal(), mConn->getNumFLocal());
      
#ifdef PV_USE_CUDNN
      krRecvPost->permuteGSynCudnnToPV(mConn->getChannel());
#endif
   }
};
} // Namespace PV

#endif // PV_USE_CUDA
#endif // _DELIVER_PUSH_CUDA_H
