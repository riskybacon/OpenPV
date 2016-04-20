#ifndef DELIVER_PUSH_H_
#define DELIVER_PUSH_H_

#include <type_traits>
#include "connections/HyPerConn.hpp"
#include "layers/HyPerLayer.hpp"
#include "include/pv_common.h"
#include "include/pv_types.h"
#include "components/Deliver.h"
#include "components/ThreadBuffer.h"

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
class DeliverPush : public Deliver {

   // Future implementation may allow for each of these types to be different.
   // These typedefs allow for a path forward with that implementation
   typedef T ActivityType;
   typedef T WeightType;
   typedef T PreType;
   typedef T PostType;

   // HyPerConn for whom propogation will be perfored
   HyPerConn *mConn;
   // Pre-synaptic layer
   const HyPerLayer * const mPre;
   // Post-synaptic layer
   HyPerLayer * mPost;
   // Number of threads that will be used
   int mNumThreads;
   // Number of batches
   int mNumBatches;
   // The accumulator function
   AccumulatorType mAccumulator;

#ifdef PV_USE_OPENMP_THREADS
   // Per-thread accumulation buffer
   ThreadBuffer<T> mThreadGSyn;
#endif // PV_USE_OPENMP_THREADS

public:
   DeliverPush(HyPerConn * conn, const HyPerLayer * const pre, HyPerLayer * post, int numThreads, int numBatches)
   : mConn(conn)
   , mPre(pre)
   , mPost(post)
   , mNumThreads(numThreads)
   , mNumBatches(numBatches)
   {
      allocateThreadGSyn(numThreads, mPost->getNumNeurons());
   }

   void operator()(PVLayerCube const * activity, int arbor, int* numActive = NULL, int** activeList = NULL) {
      // Get number of neurons restricted target
      const int numPostRestricted = mPost->getNumNeurons();

      ActivityType dtFactor = mConn->getConvertToRateDeltaTimeFactor();
      if (mConn->getPvpatchAccumulateType() == ACCUMULATE_STOCHASTIC) {
         dtFactor = mConn->getParent()->getDeltaTime();
      }

      const PVLayerLoc *preLoc = mPre->getLayerLoc();
      const PVLayerLoc *postLoc = mPost->getLayerLoc();

      const int preNx = preLoc->nx;
      const int preNy = preLoc->ny;
      const int preNf = preLoc->nf;
      const int postNx = postLoc->nx;
      const int postNy = postLoc->ny;
      const int postNf = postLoc->nf;

      const PVHalo * preHalo = &preLoc->halo;

      // The start of the gsyn buffer
      PostType * gSynPatchHead = mPost->getChannel(mConn->getChannel());

      const int numExtended = activity->numItems;

      for (int b = 0; b < mNumBatches; b++) {
         int batchOffset = b * (preNx + preHalo->rt + preHalo->lt) * (preNy + preHalo->up + preHalo->dn) * preNf;
         ActivityType * activityBatch = activity->data + batchOffset;
         PostType * gSynPatchHeadBatch = gSynPatchHead + b * postNx * postNy * postNf;
         unsigned int * activeIndicesBatch = NULL;
         int numLoop = numExtended;

         if (activity->isSparse) {
            activeIndicesBatch = activity->activeIndices + batchOffset;
            numLoop = activity->numActive[b];
         }

         clearThreadGSyn();

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int idx = 0; idx < numLoop; idx++) {
            int preExt = idx;
            if (activity->isSparse) {
               preExt = activeIndicesBatch[idx];
            }

            ActivityType a = activityBatch[preExt] * dtFactor;
            if (a == 0.0f) continue;

            PostType *gSynPatchHead = patchHead(gSynPatchHeadBatch);

            PVPatch *weights = mConn->getWeights(preExt, arbor);
            const int nk = weights->nx * mConn->fPatchSize();
            const int ny = weights->ny;
            const int sy  = mConn->getPostNonextStrides()->sy;       // stride in layer
            const int syw = mConn->yPatchStride();                   // stride in patch
            WeightType * weightDataStart = mConn->get_wData(arbor, preExt);
            PostType * postPatchStart = gSynPatchHead + mConn->getGSynPatchStart(preExt, arbor);
            for (int y = 0; y < ny; y++) {
               mAccumulator(0, nk, postPatchStart + y * sy, a, weightDataStart + y * syw, 0, dtFactor);
            }
         }

         reduceIntoPost(gSynPatchHeadBatch);
         // TODO 2016-04-20 jbowles, remove the following reduction
//#ifdef PV_USE_OPENMP_THREADS
//#pragma omp parallel for
//         for(int ni = 0; ni < mPost->getNumNeurons(); ni++){
//            for(int ti = 0; ti < mConn->getParent()->getNumThreads(); ti++){
//               gSynPatchHeadBatch[ni] += mThreadGSyn[ti][ni];
//            }
//         }
//#endif
      }
   }

private:

#ifdef PV_USE_OPENMP_THREADS
   // OpenMP Specific functions, the non-OpenMP version do nothing
   // and are compiled out in non-OpenMP builds
   /**
    * Allocate a per-thread gSyn buffer. This gives each thread a separate place to accumulate output
    */
   void allocateThreadGSyn(int numThreads, int numPostNeurons) {
      mThreadGSyn.resize(numThreads, numPostNeurons);
   }

   /** 
    * Clear the thread gSyn buffers
    */
   void clearThreadGSyn() {
      mThreadGSyn.reset();
   }

   /**
    * Reduce the thread gSyn buffers into the post layer
    */
   void reduceIntoPost(PostType * gSyn) {
      mThreadGSyn.reduce(gSyn);
   }

   /**
    * @returns the appropriate mThreadGyn for the OpenMP implementation.
    *          In the non-OpenMP implementation, this returns default head
    */
   pvdata_t *patchHead(PostType * defaultHead) {
      return mThreadGSyn[omp_get_thread_num()].data();
   }

#else // PV_USE_OPENMP_THREADS

   // Compiled out in non-OpenMP builds
   void allocateThreadGSyn(int numThreads, int numPostNeurons) {}
   void clearThreadGSyn() {}
   void reduceIntoPost(PostType * gSyn) {}

   // Returns the input in non-OpenMP build, returns pointer into
   // mThreadGSyn in OpenMP build
   pvdata_t *patchHead(PostType * defaultHead) const {
      return defaultHead;
   }
#endif // PV_USE_OPENMP_THREADS

};

} // Namespace PV

#endif // _DELIVER_PUSH_H
