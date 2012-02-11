/*
 * IncrementLayer.cpp
 *
 *  Created on: Feb 7, 2012
 *      Author: pschultz
 */

#include "IncrementLayer.hpp"

namespace PV {

IncrementLayer::IncrementLayer() {
   initialize_base();
}

IncrementLayer::IncrementLayer(const char* name, HyPerCol * hc, int numChannels) {
   initialize_base();
   initialize(name, hc, numChannels);
}

IncrementLayer::~IncrementLayer() {
   free(Vprev);
}

int IncrementLayer::initialize_base() {
   Vprev = NULL;
   displayPeriod = 0;
   VInited = false;
   nextUpdateTime = 0;
   return PV_SUCCESS;
}

int IncrementLayer::initialize(const char* name, HyPerCol * hc, int numChannels) {
   int status = ANNLayer::initialize(name, hc, numChannels);
   displayPeriod = parent->parameters()->value(name, "displayPeriod", parent->getDeltaTime());
   firstUpdateTime = parent->parameters()->value(name, "firstUpdateTime", parent->simulationTime());
   nextUpdateTime = firstUpdateTime+displayPeriod;
   Vprev = (pvdata_t *) calloc(getNumNeurons(),sizeof(pvdata_t));
   if( Vprev == NULL ) {
      fprintf(stderr, "Unable to allocate Vprev buffer for IncrementLayer \"%s\"\n", name);
      abort();
   }

   return status;
}

int IncrementLayer::readVThreshParams(PVParams * params) {
   // Threshold paramaters are not used, as updateState does not call applyVMax or applyVThresh
   // Override ANNLayer::readVThreshParams so that params file does not attempt to read the
   // threshold params
   VMax = max_pvdata_t;
   VThresh = -max_pvdata_t;
   VMin = VThresh;
   return PV_SUCCESS;
}

int IncrementLayer::updateState(float timef, float dt) {
   int status = PV_SUCCESS;
   if( !VInited && timef >= firstUpdateTime ) {
      status = updateV();
      VInited = true;
   }
   else if( VInited && timef >= nextUpdateTime ) {
      nextUpdateTime += displayPeriod;
      pvdata_t * Vprev1 = Vprev;
      pvdata_t * V = getV();
      for( int k=0; k<getNumNeurons(); k++ ) {
         *(Vprev1++) = *(V++);
      }
      updateV();
      setActivity();
   }
   resetGSynBuffers();
   return status;
}

int IncrementLayer::setActivity() {
   int status = PV_SUCCESS;
   for( int k=0; k<getNumNeurons(); k++ ) {
      clayer->activity->data[k] = getV()[k]-Vprev[k];
   }
   return status;
}

} /* namespace PV */
