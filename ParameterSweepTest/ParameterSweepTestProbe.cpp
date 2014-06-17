/*
 * ParameterSweepTestProbe.cpp
 *
 *  Created on: Aug 13, 2012
 *      Author: pschultz
 */

#include "ParameterSweepTestProbe.hpp"

namespace PV {

ParameterSweepTestProbe::ParameterSweepTestProbe(const char * probeName, HyPerCol * hc) {
   initParameterSweepTestProbe(probeName, hc);
}

ParameterSweepTestProbe::~ParameterSweepTestProbe() {
}

int ParameterSweepTestProbe::initParameterSweepTestProbe(const char * probeName, HyPerCol * hc) {
   int status = initStatsProbe(probeName, hc);
   return status;
}

int ParameterSweepTestProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = StatsProbe::ioParamsFillGroup(ioFlag);
   ioParam_expectedSum(ioFlag);
   ioParam_expectedMin(ioFlag);
   ioParam_expectedMax(ioFlag);
   return status;
}

void ParameterSweepTestProbe::ioParam_buffer(enum ParamsIOFlag ioFlag) {
   requireType(BufActivity);
}

void ParameterSweepTestProbe::ioParam_expectedSum(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "expectedSum", &expectedSum, 0.0);
}
void ParameterSweepTestProbe::ioParam_expectedMin(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "expectedMin", &expectedMin, 0.0f);
}

void ParameterSweepTestProbe::ioParam_expectedMax(enum ParamsIOFlag ioFlag) {
   getParent()->ioParamValue(ioFlag, getName(), "expectedMax", &expectedMax, 0.0f);
}

int ParameterSweepTestProbe::outputState(double timed) {
   int status = StatsProbe::outputState(timed);
#ifdef PV_USE_MPI
   InterColComm * icComm = getTargetLayer()->getParent()->icCommunicator();
   const int rcvProc = 0;
   if( icComm->commRank() != rcvProc ) {
      return 0;
   }
#endif // PV_USE_MPI
   if (timed >= 3.0 ) {
      assert(fabs(expectedSum - sum)<1e-6);
      assert(fabs(expectedMin - fMin)<1e-6);
      assert(fabs(expectedMax - fMax)<1e-6);
   }
   return status;
}

} /* namespace PV */
