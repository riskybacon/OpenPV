/*
 * KernelPactchProbe.cpp
 *
 *  Created on: Oct 21, 2011
 *      Author: pschultz
 */

#include "KernelProbe.hpp"

namespace PV {

KernelProbe::KernelProbe() {
   initialize_base();
}

KernelProbe::KernelProbe(const char * probename, HyPerCol * hc) {
   initialize_base();
   int status = initialize(probename, hc);
   assert(status == PV_SUCCESS);
}

KernelProbe::~KernelProbe() {
}

int KernelProbe::initialize_base() {
   return PV_SUCCESS;
}

int KernelProbe::initialize(const char * probename, HyPerCol * hc) {
   int status = BaseConnectionProbe::initialize(probename, hc);
   assert(name && parent);

   return status;
}

int KernelProbe::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = BaseConnectionProbe::ioParamsFillGroup(ioFlag);
   ioParam_kernelIndex(ioFlag);
   ioParam_arborId(ioFlag);
   ioParam_outputWeights(ioFlag);
   ioParam_outputPlasticIncr(ioFlag);
   ioParam_outputPatchIndices(ioFlag);
   return status;
}

void KernelProbe::ioParam_kernelIndex(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "kernelIndex", &kernelIndex, 0);
}

void KernelProbe::ioParam_arborId(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "arborId", &arborID, 0);
}

void KernelProbe::ioParam_outputWeights(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "outputWeights", &outputWeights, true/*default value*/);
}

void KernelProbe::ioParam_outputPlasticIncr(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "outputPlasticIncr", &outputPlasticIncr, false/*default value*/);
}

void KernelProbe::ioParam_outputPatchIndices(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "outputPatchIndices", &outputPatchIndices, false/*default value*/);
}

int KernelProbe::communicate() {
   int status = PV_SUCCESS;
   assert(targetConn);
   if(getTargetConn()->usingSharedWeights()==false) {
      fprintf(stderr, "KernelProbe \"%s\": connection \"%s\" is not using shared weights.\n", name, targetConn->getName());
      status = PV_FAILURE;
   }
#ifdef PV_USE_MPI
   MPI_Barrier(parent->icCommunicator()->communicator());
#endif
   if (status != PV_SUCCESS) {
      exit(EXIT_FAILURE);
   }
   return status;
}

int KernelProbe::allocateDataStructures() {
   int status = PV_SUCCESS;
   assert(getTargetConn());
   if (getKernelIndex()<0 || getKernelIndex()>=getTargetConn()->getNumDataPatches()) {
      fprintf(stderr, "KernelProbe \"%s\": kernelIndex %d is out of bounds.  (min 0, max %d)\n", name, getKernelIndex(), getTargetConn()->getNumDataPatches()-1);
      exit(EXIT_FAILURE);
   }
   if (getArbor()<0 || getArbor()>=getTargetConn()->numberOfAxonalArborLists()) {
      fprintf(stderr, "KernelProbe \"%s\": arborId %d is out of bounds. (min 0, max %d)\n", name, getArbor(), getTargetConn()->numberOfAxonalArborLists()-1);
      exit(EXIT_FAILURE);
   }

   if(outputstream) {
      fprintf(outputstream->fp, "Probe \"%s\", kernel index %d, arbor index %d.\n", name, getKernelIndex(), getArbor());
   }
   if(getOutputPatchIndices()) {
      patchIndices(getTargetConn());
   }

   return status;
}

int KernelProbe::outputState(double timed) {
#ifdef PV_USE_MPI
   InterColComm * icComm = parent->icCommunicator();
   const int rank = icComm->commRank();
   if( rank != 0 ) return PV_SUCCESS;
#endif // PV_USE_MPI
   assert(getTargetConn()!=NULL);
   int nxp = getTargetConn()->xPatchSize();
   int nyp = getTargetConn()->yPatchSize();
   int nfp = getTargetConn()->fPatchSize();
   int patchSize = nxp*nyp*nfp;

   const pvwdata_t * wdata = getTargetConn()->get_wDataStart(arborID)+patchSize*kernelIndex;
   const pvwdata_t * dwdata = outputPlasticIncr ?
         getTargetConn()->get_dwDataStart(arborID)+patchSize*kernelIndex : NULL;
   fprintf(outputstream->fp, "Time %f, Conn \"%s\", nxp=%d, nyp=%d, nfp=%d\n",
           timed, getTargetConn()->getName(),nxp, nyp, nfp);
   for(int f=0; f<nfp; f++) {
      for(int y=0; y<nyp; y++) {
         for(int x=0; x<nxp; x++) {
            int k = kIndex(x,y,f,nxp,nyp,nfp);
            fprintf(outputstream->fp, "    x=%d, y=%d, f=%d (index %d):", x, y, f, k);
            if(getOutputWeights()) {
               fprintf(outputstream->fp, "  weight=%f", (float)wdata[k]);
            }
            if(getOutputPlasticIncr()) {
               fprintf(outputstream->fp, "  dw=%f", (float)dwdata[k]);
            }
            fprintf(outputstream->fp,"\n");
         }
      }
   }

   return PV_SUCCESS;
}

int KernelProbe::patchIndices(HyPerConn * conn) {
   int nxp = conn->xPatchSize();
   int nyp = conn->yPatchSize();
   int nfp = conn->fPatchSize();
   int nPreExt = conn->getNumWeightPatches();
   assert(nPreExt == conn->preSynapticLayer()->getNumExtended());
   const PVLayerLoc * loc = conn->preSynapticLayer()->getLayerLoc();
   int marginWidth = loc->nb;
   int nxPre = loc->nx;
   int nyPre = loc->ny;
   int nfPre = loc->nf;
   int nxPreExt = nxPre+2*marginWidth;
   int nyPreExt = nyPre+2*marginWidth;
   for( int kPre = 0; kPre < nPreExt; kPre++ ) {
      PVPatch * w = conn->getWeights(kPre,arborID);
      int xOffset = kxPos(w->offset, nxp, nyp, nfp);
      int yOffset = kyPos(w->offset, nxp, nyp, nfp);
      int kxPre = kxPos(kPre,nxPreExt,nyPreExt,nfPre)-marginWidth;
      int kyPre = kyPos(kPre,nxPreExt,nyPreExt,nfPre)-marginWidth;
      int kfPre = featureIndex(kPre,nxPreExt,nyPreExt,nfPre);
      fprintf(outputstream->fp,"    presynaptic neuron %d (x=%d, y=%d, f=%d) uses kernel index %d, starting at x=%d, y=%d\n",
            kPre, kxPre, kyPre, kfPre, conn->patchIndexToDataIndex(kPre), xOffset, yOffset);
   }
   return PV_SUCCESS;
}

}  // end of namespace PV block
