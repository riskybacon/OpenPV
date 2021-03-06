/*
 * CIFARGTLayer.cpp
 * Author: slundquist
 */

#include "CIFARGTLayer.hpp"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>

namespace PV {
CIFARGTLayer::CIFARGTLayer(const char * name, HyPerCol * hc)
{
   initialize(name, hc);
}

CIFARGTLayer::~CIFARGTLayer(){
   if(inputfile) inputfile.close();
}


int CIFARGTLayer::initialize(const char * name, HyPerCol * hc) {
   //TODO make only root process do this
   //Is there a way to implement a test for mpi?
   int status = ANNLayer::initialize(name, hc);
   negativeGt = true;
   //2 files are test and train, assuming name of the layer is either test or train
   //std::string filename = "input/" + std::string(name) + ".txt";
   inputfile.open(inFilename, std::ifstream::in);
   if (!inputfile.is_open()){
      std::cout << "Unable to open file " << inFilename << "\n";
      exit(EXIT_FAILURE);
   }
   if(startFrame < 1){
      std::cout << "Setting startFrame to 1\n";
      startFrame = 1;
   }
   //Skip for startFrame
   for(int i = 0; i < startFrame; i++){
      getline (inputfile,inputString);
   }
   return status;
}

int CIFARGTLayer::ioParamsFillGroup(enum ParamsIOFlag ioFlag) {
   int status = ANNLayer::ioParamsFillGroup(ioFlag);
   ioParam_inFilename(ioFlag);
   ioParam_StartFrame(ioFlag);
   ioParam_NegativeGt(ioFlag);
   return status;
}

void CIFARGTLayer::ioParam_inFilename(enum ParamsIOFlag ioFlag) {
   parent->ioParamStringRequired(ioFlag, name, "inFilename", &inFilename);
}

void CIFARGTLayer::ioParam_StartFrame(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "startFrame", &startFrame, startFrame);
}

void CIFARGTLayer::ioParam_NegativeGt(enum ParamsIOFlag ioFlag) {
   parent->ioParamValue(ioFlag, name, "negativeGt", &negativeGt, negativeGt);
}

int CIFARGTLayer::updateState(double timef, double dt) {
   pvdata_t * A = getCLayer()->activity->data;
   const PVLayerLoc * loc = getLayerLoc(); 
   
   getline (inputfile,inputString);
   unsigned found = inputString.find_last_of("/\\");
   //CIFAR is 0 indexed
   char cVal = inputString.at(found-1);
   int iVal = cVal - '0';
   std::cout << "time: " << parent->simulationTime() << " inputString:" << inputString << "  iVal:" << iVal << "\n";
   assert(iVal >= 0 && iVal < 10);
   //NF must be 10, one for each class
   assert(loc->nf == 10);
   for(int ni = 0; ni < getNumNeurons(); ni++){
      int nExt = kIndexExtended(ni, loc->nx, loc->ny, loc->nf, loc->halo.lt, loc->halo.rt, loc->halo.dn, loc->halo.up);
      int fi = featureIndex(nExt, loc->nx+loc->halo.rt+loc->halo.lt, loc->ny+loc->halo.dn+loc->halo.up, loc->nf);
      if(fi == iVal){
         A[nExt] = 1;
      }
      else{
         if(negativeGt){
            A[nExt] = -1;
         }
         else{
            A[nExt] = 0;
         }
      }
   }
   return PV_SUCCESS;
}
}
