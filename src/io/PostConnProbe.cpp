/*
 * PostConnProbe.cpp
 *
 *  Created on: May 12, 2009
 *      Author: rasmussn
 */

#include "PostConnProbe.hpp"
#include <assert.h>

namespace PV {

/**
 * @kPost
 */
PostConnProbe::PostConnProbe(int kPost)
   : ConnectionProbe(0)
{
   this->kxPost = 0;
   this->kyPost = 0;
   this->kfPost = 0;
   this->kPost = kPost;
}

/**
 * @filename
 * @kPost
 */
PostConnProbe::PostConnProbe(const char * filename, int kPost)
   : ConnectionProbe(filename, 0)
{
   this->kxPost = 0;
   this->kyPost = 0;
   this->kfPost = 0;
   this->kPost = kPost;
}

PostConnProbe::PostConnProbe(int kxPost, int kyPost, int kfPost)
   : ConnectionProbe(0)
{
   this->kxPost = kxPost;
   this->kyPost = kyPost;
   this->kfPost = kfPost;
   this->kPost = -1;
}

PostConnProbe::PostConnProbe(const char * filename,int kxPost, int kyPost, int kfPost)
   : ConnectionProbe(filename, 0, 0, 0)
{
   this->kxPost = kxPost;
   this->kyPost = kyPost;
   this->kfPost = kfPost;
   this->kPost = -1;
}
/**
 * @time
 * @c
 * NOTES:
 *    - kPost , kxPost, kyPost are indices in the restricted post-synaptic layer.
 *
 */
int PostConnProbe::outputState(float time, HyPerConn * c)
{
   int kxPre, kyPre;
   PVPatch  * w;
   PVPatch ** wPost = c->convertPreSynapticWeights(time);

   const PVLayer * l = c->postSynapticLayer()->clayer;

   const int nx = l->loc.nx;
   const int ny = l->loc.ny;
   const int nf = l->numFeatures;

   // calc kPost if needed
   if (kPost < 0) {
      kPost = kIndex(kxPost, kyPost, kfPost, nx, ny, nf);
   }
   else {
      kxPost = kxPos(kPost, nx, ny, nf);
      kyPost = kyPos(kPost, nx, ny, nf);
      kfPost = featureIndex(kPost, nx, ny, nf);
   }

   c->preSynapticPatchHead(kxPost, kyPost, kfPost, &kxPre, &kyPre);

   w = wPost[kPost];

   fprintf(fp, "w%d(%d,%d,%d) prePatchHead(%d,%d): ", kPost, kxPost, kyPost, kfPost, kxPre, kyPre);

   text_write_patch(fp, w, w->data);
   fflush(fp);

   if (outputIndices) {
      const PVLayer * lPre = c->preSynapticLayer()->clayer;
      write_patch_indices(fp, w, &lPre->loc, kxPre, kyPre, 0);
      fflush(fp);
   }

   return 0;
}

} // namespace PV
