/*
 * 3STDPConn.hpp
 *
 *  Created on: July 24, 2012
 *      Author: Rui P. Costa
 */

#ifndef STDP3CONN_HPP_
#define STDP3CONN_HPP_

#include "HyPerConn.hpp"
#include "../include/default_params.h"
#include <stdio.h>

namespace PV {

class STDP3Conn : HyPerConn {
public:
   STDP3Conn(const char * name, HyPerCol * hc);
   virtual ~STDP3Conn();

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers(const char * kernelName);
   virtual int initializeThreadKernels(const char * kernelName);
#endif // PV_USE_OPENCL

   virtual int allocateDataStructures();

   // virtual int deleteWeights(); // Changed to a private method.  Should not be virtual since it's called from the destructor.

   virtual float maxWeight(int arborId = 0);

   virtual int updateState(double time, double dt);
   virtual int updateWeights(int axonId);
   virtual int outputState(double time, bool last=false);
   virtual int writeTextWeightsExtra(PV_Stream * pvstream, int k, int arborID);

   virtual PVLayerCube * getPlasticityDecrement();

protected:
   STDP3Conn();
   int initialize_base();
   int initialize(const char * name, HyPerCol * hc);
   virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
   virtual void ioParam_stdpFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_ampLTP(enum ParamsIOFlag ioFlag);
   virtual void ioParam_ampLTD(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauLTP(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauLTD(enum ParamsIOFlag ioFlag);
   virtual void ioParam_tauY(enum ParamsIOFlag ioFlag);
   virtual void ioParam_wMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_wMin(enum ParamsIOFlag ioFlag);
   virtual void ioParam_dwMax(enum ParamsIOFlag ioFlag);
   virtual void ioParam_synscalingFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_synscaling_v(enum ParamsIOFlag ioFlag);

   virtual int initPlasticityPatches();

   PVLayerCube    * post_tr;      // plasticity decrement variable for postsynaptic layer
   PVLayerCube    * post2_tr;      // second plasticity decrement variable for postsynaptic layer
   PVLayerCube    * pre_tr;      // plasticity increment variable for presynaptic layer

   bool       stdpFlag;         // presence of spike timing dependent plasticity

   int pvpatch_update_plasticity_incr(int nk, float * RESTRICT p,
                                      float aj, float decay, float fac);
   int pvpatch_update_weights(int nk, float * RESTRICT w, const float * RESTRICT m,
                              const float * RESTRICT p, float aPre,
                              const float * RESTRICT aPost, float dWmax, float wMin, float wMax);

   // STDP parameters for modifying weights
   float ampLTP; // long term potentiation amplitude
   float ampLTD; // long term depression amplitude
   float tauLTP;
   float tauLTD;
   float tauY;
   float dWMax;
   bool synscalingFlag;
   float synscaling_v;

private:
   int deleteWeights();

};

}

#endif /* STDPCONN_HPP_ */
