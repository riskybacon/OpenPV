#ifndef DELIVER_HPP_
#define DELIVER_HPP_

#include <type_traits>
#include "connections/HyPerConn.hpp"
#include "layers/HyPerLayer.hpp"
#include "include/pv_common.h"
#include "include/pv_types.h"

namespace PV {

/**
 * Base delivery class. Delivers the pre-synaptic perspective to the post synaptic
 * layer. The base class does very little and should not be used directly.
 * Instead, create a subclasss, preferably a subclass that is templatized for
 * data types, gSyn accumulation types, patch geometries, accumlation functors, etc.
 */
class Deliver {
public:
   Deliver() {}
   virtual void operator()(const PVLayerCube * const activity, int arbor) = 0;
};

// Todo:
//
// This class has too much knowledge about HyPerConn, which makes this class pretty un-reusable
// outside of HyPerConn.
//
// Right now, all this class provides is a way to pull delivery methods out of HyPerConn and
// allow for inlining of the inner convolution. Testing shows that this results in a 20% performance
// increase, so the change is worth it just for that.
//
// It also allows for the easy addition of new delivery methods without cluttering up HyPerConn.
template<typename AccumulatorType,
   typename ActivityType = typename AccumulatorType::Type,
   typename WeightType = typename AccumulatorType::Type,
   typename PreType = typename AccumulatorType::Type,
   typename PostType = typename AccumulatorType::Type>
class DeliverPre : public Deliver {
private:
   PV::HyPerConn *mConn;
   int mNfp;
   const HyPerLayer * const mPre;
   HyPerLayer * mPost;
   int mNumThreads;
   int mNumBatches;
   AccumulatorType mAccumulator;

public:
   DeliverPre(HyPerConn * conn, HyPerLayer * pre, HyPerLayer * post, int numThreads, int numBatches)
   : mConn(conn)
   , mPre(pre)
   , mPost(post)
   , mNumThreads(numThreads)
   , mNumBatches(numBatches)
   {
      allocateThreadGSyn(numThreads, mPost->getNumNeurons());
   }

   void operator()(const PVLayerCube * const activity, int arbor) {
      ActivityType dtFactor = mConn->getConvertToRateDeltaTimeFactor();
      if (mConn->getPvpatchAccumulateType() == ACCUMULATE_STOCHASTIC) {
         dtFactor = mConn->getParent()->getDeltaTime();
      }

#if 0
      // Belongs in DeliverPreSparse
      if (_weightSparsity > 0.0f && !_sparseWeightsAllocated[arbor]) {
         allocateSparseWeightsPre(activity, arbor);
      }
#endif

      const PVLayerLoc *preLoc = mPre->getLayerLoc();
      const PVLayerLoc *postLoc = mPost->getLayerLoc();

      const int numExtended = activity->numItems;
      int numPostNeurons = mPost->getNumNeurons();

      for (int b = 0; b < mNumBatches; b++) {
         int batchOffset = b * (preLoc->nx + preLoc->halo.rt + preLoc->halo.lt) * (preLoc->ny + preLoc->halo.up + preLoc->halo.dn) * preLoc->nf;
         pvdata_t * activityBatch = activity->data + batchOffset;
         pvdata_t * gSynPatchHeadBatch = mPost->getChannel(mConn->getChannel()) + b * postLoc->nx * postLoc->ny * postLoc->nf;
         unsigned int * activeIndicesBatch = NULL;
         int numLoop = numExtended;

         if (activity->isSparse) {
            activeIndicesBatch = activity->activeIndices + batchOffset;
            numLoop = activity->numActive[b];
         }

         clearThreadGSyn(mNumThreads, numPostNeurons);

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
         for (int idx = 0; idx < numLoop; idx++) {
            int kPreExt = idx;
            if (activity->isSparse) {
               kPreExt = activeIndicesBatch[idx];
            }

            float a = activityBatch[kPreExt] * dtFactor;
            if (a == 0.0f) continue;

            pvdata_t *gSynPatchHead = patchHead(gSynPatchHeadBatch);

            PVPatch *weights = mConn->getWeights(kPreExt, arbor);
            const int nk = weights->nx * mConn->fPatchSize();
            const int ny = weights->ny;
            const int sy  = mConn->getPostNonextStrides()->sy;       // stride in layer
            const int syw = mConn->yPatchStride();                   // stride in patch
            WeightType * weightDataStart = mConn->get_wData(arbor, kPreExt);
            PostType * postPatchStart = gSynPatchHead + mConn->getGSynPatchStart(kPreExt, arbor);
            for (int y = 0; y < ny; y++) {
               mAccumulator(0, nk, postPatchStart + y * sy, a, weightDataStart + y * syw, 0, dtFactor);
            }
         }
         
         accumulateIntoPost(gSynPatchHeadBatch, numPostNeurons);
      }
   }

private:

#ifdef PV_USE_OPENMP_THREADS
   // OpenMP Specific functions, the non-OpenMP version do nothing
   // and are compiled out in non-OpenMP builds

   // Accumulation buffer for each thread
   std::vector< std::vector<PostType> > mThreadGSyn;

   /**
    * Allocate a per-thread gSyn buffer. This gives each thread a separate place to accumulate output
    */
   void allocateThreadGSyn(int numThreads, int numPostNeurons) {
      mThreadGSyn.resize(numThreads);

      for (auto& gSyn : mThreadGSyn) {
         gSyn.resize(numPostNeurons);
      }
   }

   void clearThreadGSyn(int numThreads, int numPostNeurons) {
      //#pragma omp parallel for
      for (int i = 0; i < numThreads * numPostNeurons; i++) {
         int ti = i / numPostNeurons;
         int ni = i % numPostNeurons;
         mThreadGSyn[ti][ni] = 0;
      }
   }

   /**
    * Reduce the thread gSyn buffers into the post layer
    */
   void accumulateIntoPost(pvdata_t *gSyn, int numPostNeurons) {
      // Memory access patterns may be inefficient here. However, there are
      // no collisions when updating the post synaptic perspective, so no
      // atomics or locking required.
#pragma omp parallel for
      for(int ni = 0; ni < numPostNeurons; ni++){
         for(int ti = 0; ti < mNumThreads; ti++){
            gSyn[ni] += mThreadGSyn[ti][ni];
         }
      }
   }

   /**
    * @returns the appropriate mThreadGyn for the OpenMP implementation.
    *          In the non-OpenMP implementation, this returns default head
    */
   pvdata_t *patchHead(pvdata_t *defaultHead) {
      Debug() << omp_get_thread_num() << "/" << omp_get_num_threads() << std::endl;
      return mThreadGSyn[omp_get_thread_num()].data();
   }

#else // PV_USE_OPENMP_THREADS

   // Compiled out in non-OpenMP builds
   void allocateThreadGSyn(int numThreads, int numPostNeurons) {}
   void clearThreadGSyn(int numThreads, int numPostNeurons) {}
   void accumulateIntoPost(pvdata_t *gSyn) {}

   // Returns the input in non-OpenMP build, returns pointer into
   // mThreadGSyn in OpenMP build
   pvdata_t *patchHead(pvdata_t *defaultHead) const {
      return defaultHead;
   }
#endif // PV_USE_OPENMP_THREADS

};

} // Namespace PV

#endif // _DELIVER_HPP

