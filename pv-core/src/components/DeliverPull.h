#ifndef DELIVER_PULL_H_
#define DELIVER_PULL_H_

#include <type_traits>
#include "connections/HyPerConn.hpp"
#include "layers/HyPerLayer.hpp"
#include "include/pv_common.h"
#include "include/pv_types.h"
#include "components/Deliver.h"

namespace PV {

   /**
    * Pull the pre-synaptic perspective over the synapse and onto the post-synaptic perspective.
    *
    * The pull method of delivery has an advantage over the push method because each target
    * neuron can be updated on separate threads.
    *
    * A downside to the pull method is that it currently cannot use activation sparsity
    * like the push method. Use the pull method when activation sparsity is not being used.
    *
    * ISSUES
    *
    * This class should not be dependent on HyPerConn. This creates a circular class
    * dependency and indicates that HyPerConn has grown too large.
    *
    * Dependencies
    *
    * - HyPerLayer: source and destination neurons)
    * - HyPerConn: weight values and data structure geometry, dtFactor
    * - PVLayerCube: source and destination neurons
    */
template<typename AccumulatorType, typename T = float>
class DeliverPull : public Deliver {

   // Future implementation may allow for each of these types to be different.
   // These typedefs allow for a path forward with that implementation
   typedef T ActivityType;
   typedef T WeightType;
   typedef T PreType;
   typedef T PostType;

   HyPerConn *mConn;
   HyPerConn *mPostConn;
   const HyPerLayer * const mPre;
   HyPerLayer * mPost;
   int mNumThreads;
   int mNumBatches;
   AccumulatorType mAccumulator;
   int mSyp;
   int mYPatchSize;
   int mNumPerStride;
   int mKernelIndex;
public:
   DeliverPull(HyPerConn * conn, const HyPerLayer * const pre, HyPerLayer * post, int numThreads, int numBatches)
   : mConn(conn)
   , mPostConn(conn->getPostConn())
   , mPre(pre)
   , mPost(post)
   , mNumThreads(numThreads)
   , mNumBatches(numBatches)
   , mSyp(conn->getPostConn()->yPatchStride())
   , mYPatchSize(conn->getPostConn()->yPatchSize())
   , mNumPerStride(conn->getPostConn()->xPatchSize() * conn->getPostConn()->fPatchSize())
   {
   }

   void operator()(PVLayerCube const * activity, int arbor, int* numActive = NULL, int** activeList = NULL) {
      // Get number of neurons restricted target
      const int numPostRestricted = mPost->getNumNeurons();

      ActivityType dtFactor = mConn->getConvertToRateDeltaTimeFactor();
      if (mConn->getPvpatchAccumulateType() == ACCUMULATE_STOCHASTIC) {
         dtFactor = mConn->getParent()->getDeltaTime();
      }

      const PVLayerLoc * preLoc = mPre->getLayerLoc();
      const PVLayerLoc * postLoc = mPost->getLayerLoc();

      const int preNx = preLoc->nx;
      const int preNy = preLoc->ny;
      const int preNf = preLoc->nf;
      const int postNx = postLoc->nx;
      const int postNy = postLoc->ny;
      const int postNf = postLoc->nf;

      const PVHalo * preHalo = &preLoc->halo;
      const PVHalo * postHalo = &postLoc->halo;

      // Get source layer's extended y stride
      int sy = (preNx + preHalo->lt + preHalo->rt) * preNf;

      // The start of the gsyn buffer
      PostType * gSynPatchHead = mPost->getChannel(mConn->getChannel());

      long * startPreExtBuf = mConn->getPostToPreActivity();
      if (!startPreExtBuf) {
         std::cout << "HyPerLayer::recvFromPost error getting preToPostActivity from connection. Is shrink_patches on?\n";
         exit(EXIT_FAILURE);
      }

      // If numActive is a valid pointer, we're recv from post sparse
      bool recvPostSparse = numActive;

      for(int b = 0; b < mNumBatches; b++) {
         int batchOffset = b * (preNx + preHalo->rt + preHalo->lt) * (preNy + preHalo->up + preHalo->dn) * preNf;
         ActivityType * activityBatch = activity->data + batchOffset;
         PostType * gSynPatchHeadBatch = gSynPatchHead + b * postNx * postNy * postNf;

         int numLoop = numPostRestricted;
         if (recvPostSparse) {
            numLoop = numActive[b];
         }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for schedule(static)
#endif
         for (int idx = 0; idx < numLoop; idx++) {
            int postRes = idx;
            if (recvPostSparse) {
               postRes = activeList[b][idx];
            }
            // Change restricted to extended post neuron
            int targetExt = kIndexExtended(postRes, postNx, postNy, postNf, postHalo->lt, postHalo->rt, postHalo->dn, postHalo->up);

            // Read from buffer
            long startPreExt = startPreExtBuf[postRes];

            // Calculate target's start of gsyn
            PostType * gSynPatchPos = gSynPatchHeadBatch + postRes;

            taus_uint4 * rngPtr = mConn->getRandState(postRes);
            ActivityType * activityStartBuf = &(activityBatch[startPreExt]);

            int kernelIndex = mPostConn->patchToDataLUT(targetExt);

            WeightType * weightStartBuf = mPostConn->get_wDataHead(arbor, kernelIndex);
            for (int ky = 0; ky < mYPatchSize; ky++) {
               float * activityY = &(activityStartBuf[ky * sy]);
               WeightType * weightY = weightStartBuf + ky * mSyp;
               mAccumulator(0, mNumPerStride, gSynPatchPos, activityY, weightY, rngPtr, dtFactor);
            }
         }
      }
   }
};
   
} // Namespace PV

#endif // _DELIVER_PULL_H

