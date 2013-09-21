/*
 * GPUTestForOnesProbe.cpp
 *
 *  Created on: Sep 6, 2011
 *      Author: kpeterson
 */

#include "GPUTestForOnesProbe.hpp"
#include <include/pv_arch.h>
#include <layers/HyPerLayer.hpp>
#include <string.h>
#include <assert.h>

namespace PV {

GPUTestForOnesProbe::GPUTestForOnesProbe(const char * filename, HyPerLayer * layer, const char * msg)
: StatsProbe(filename, layer, msg)
{
}

GPUTestForOnesProbe::GPUTestForOnesProbe(HyPerLayer * layer, const char * msg)
: StatsProbe(layer, msg)
{
}

GPUTestForOnesProbe::~GPUTestForOnesProbe() {}

int GPUTestForOnesProbe::outputState(double timed)
{
	int status = StatsProbe::outputState(timed);
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
	if(timed>1.0f){
		assert((fMin>0.99)&&(fMin<1.01));
		assert((fMax>0.99)&&(fMax<1.01));
		assert((avg>0.99)&&(avg<1.01));
	}

	return status;
}


} /* namespace PV */
