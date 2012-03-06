/*
 * HyPerConnection.hpp
 *
 *  Created on: Oct 21, 2008
 *      Author: Craig Rasmussen
 */

#ifndef HYPERCONN_HPP_
#define HYPERCONN_HPP_

#include "../columns/InterColComm.hpp"
#include "../include/pv_common.h"
#include "../include/pv_types.h"
#include "../io/PVParams.hpp"
#include "../io/BaseConnectionProbe.hpp"
#include "../layers/HyPerLayer.hpp"
#include "../utils/Timer.hpp"
#include "../weightinit/InitWeights.hpp"
#include <stdlib.h>

#ifdef PV_USE_OPENCL
#undef DEBUG_OPENCL  //this is used with some debug code
#include "../arch/opencl/CLKernel.hpp"
#include "../arch/opencl/CLBuffer.hpp"
#endif

#define PROTECTED_NUMBER 13
#define MAX_ARBOR_LIST (1+MAX_NEIGHBORS)

namespace PV {

class HyPerCol;
class HyPerLayer;
class InitWeights;
class InitUniformRandomWeights;
class InitGaussianRandomWeights;
class InitSmartWeights;
class InitCocircWeights;
class BaseConnectionProbe;
class PVParams;

/**
 * A HyPerConn identifies a connection between two layers
 */

class HyPerConn {

public:
   HyPerConn();
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel);
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel, const char * filename);
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel, const char * filename, InitWeights *weightInit);
   HyPerConn(const char * name, HyPerCol * hc, HyPerLayer * pre, HyPerLayer * post,
             ChannelType channel, InitWeights *weightInit);
   virtual ~HyPerConn();

   virtual int deliver(Publisher * pub, const PVLayerCube * cube, int neighbor);
#ifdef PV_USE_OPENCL
#   ifdef DEBUG_OPENCL
   virtual int deliverOpenCL(Publisher * pub, const PVLayerCube * cube);
#   else
   virtual int deliverOpenCL(Publisher * pub);
#   endif
#endif

   virtual int checkpointRead(float *timef);
   virtual int checkpointWrite();

   virtual int insertProbe(BaseConnectionProbe * p);
   virtual int outputState(float time, bool last=false);
   virtual int updateState(float time, float dt);
   virtual int updateWeights(int axonId = 0);

   virtual int writeWeights(float time, bool last=false);
   virtual int writeWeights(const char * filename);
   virtual int writeWeights(PVPatch *** patches, pvdata_t ** dataStart, int numPatches, const char * filename, float timef, bool last);
#ifdef OBSOLETE // Marked obsolete Nov 29, 2011.
   virtual int writeWeights(PVPatch ** patches, int numPatches,
                            const char * filename, float time, bool last, int arborId);
#endif // OBSOLETE_NBANDSFORARBORS
   virtual int writeTextWeights(const char * filename, int k);
   virtual int writeTextWeightsExtra(FILE * fd, int k, int arborID)
                                                    {return PV_SUCCESS;}

   virtual int writePostSynapticWeights(float time, bool last);
#ifdef OBSOLETE  // Marked obsolete Nov 29, 2011.
   virtual int writePostSynapticWeights(float time, bool last, int axonID);
#endif // OBSOLETE_NBANDSFORARBORS

   int readWeights(const char * filename);


   bool stochasticReleaseFlag;
   int (*accumulateFunctionPointer)(int nk, float* RESTRICT v, float a, float* RESTRICT w);
   // TODO make a get-method to return this.

   virtual PVLayerCube * getPlasticityDecrement()    {return NULL;}


   inline const char * getName()                     {return name;}
   inline HyPerCol * getParent()                     {return parent;}
   inline HyPerLayer * getPre()                      {return pre;}
   inline HyPerLayer * getPost()                     {return post;}
   inline ChannelType getChannel()                   {return channel;}
   inline InitWeights * getWeightInitializer()       {return weightInitializer;}
   void setDelay(int axonId, int delay);
   inline int getDelay(int arborId = 0)               {assert(arborId>=0 && arborId<numAxonalArborLists); return delays[arborId];}

   inline bool getSelfFlag(){return selfFlag;};
   virtual float minWeight(int arborId = 0);
   virtual float maxWeight(int arborId = 0);

   inline int xPatchSize()                           {return nxp;}
   inline int yPatchSize()                           {return nyp;}
   inline int fPatchSize()                           {return nfp;}
   inline int xPatchStride()                         {return sxp;}
   inline int yPatchStride()                         {return syp;}
   inline int fPatchStride()                         {return sfp;}
   inline int xPostPatchSize()                            {return nxpPost;}
   inline int yPostPatchSize()                            {return nypPost;}
   inline int fPostPatchSize()                            {return nfpPost;}

   //arbor and weight patch related get/set methods:
   inline PVPatch ** weights(int arborId = 0)        {return wPatches[arborId];}
   virtual PVPatch * getWeights(int kPre, int arborId);
   // inline PVPatch * getPlasticIncr(int kPre, int arborId) {return plasticityFlag ? dwPatches[arborId][kPre] : NULL;}
   inline pvdata_t * getPlasticIncr(int kPre, int arborId) {return plasticityFlag ? &dwDataStart[arborId][kPre*nxp*nyp*nfp + wPatches[arborId][kPre]->offset] : NULL;}
   inline const PVPatchStrides * getPostExtStrides() {return &postExtStrides;}
   inline const PVPatchStrides * getPostNonextStrides() {return &postNonextStrides;}

   inline pvdata_t * get_wDataStart(int arborId) {return wDataStart[arborId];}
   // inline void set_wDataStart(int arborId, pvdata_t * pDataStart) {wDataStart[arborId]=pDataStart;} // Should be protected
   inline pvdata_t * get_wDataHead(int arborId, int dataIndex) {return &wDataStart[arborId][dataIndex*nxp*nyp*nfp];}
   inline pvdata_t * get_wData(int arborId, int patchIndex) {return &wDataStart[arborId][patchToDataLUT(patchIndex)*nxp*nyp*nfp + wPatches[arborId][patchIndex]->offset];}

   inline pvdata_t * get_dwDataStart(int arborId) {return dwDataStart[arborId];}
   // inline void set_dwDataStart(int arborId, pvdata_t * pIncrStart) {dwDataStart[arborId]=pIncrStart;} // Should be protected
   inline pvdata_t * get_dwDataHead(int arborId, int dataIndex) {return &dwDataStart[arborId][dataIndex*nxp*nyp*nfp];}
   inline pvdata_t * get_dwData(int arborId, int patchIndex) {return &dwDataStart[arborId][patchToDataLUT(patchIndex)*nxp*nyp*nfp + wPatches[arborId][patchIndex]->offset];}

   inline PVPatch * getWPostPatches(int arbor, int patchIndex) {return wPostPatches[arbor][patchIndex];}
   inline pvdata_t * getWPostData(int arbor, int patchIndex) {return &wPostDataStart[arbor][patchIndex*nxpPost*nypPost*nfpPost]+wPostPatches[arbor][patchIndex]->offset;}
   inline pvdata_t * getWPostData(int arbor) {return wPostDataStart[arbor];}

   virtual int getNumWeightPatches();
   virtual int getNumDataPatches();
   inline  int numberOfAxonalArborLists()            {return numAxonalArborLists;}

   inline pvdata_t * getGSynPatchStart(int kPre, int arborId) {return gSynPatchStart[arborId][kPre];}
   inline size_t getAPostOffset(int kPre, int arborId) {return aPostOffset[arborId][kPre];}

   HyPerLayer * preSynapticLayer()                   {return pre;}
   HyPerLayer * postSynapticLayer()                  {return post;}

   int  getConnectionId()                            {return connId;}
   void setConnectionId(int id)                      {connId = id;}

   virtual int setParams(PVParams * params /*, PVConnParams * p*/);

   PVPatch *** convertPreSynapticWeights(float time);

   int preSynapticPatchHead(int kxPost, int kyPost, int kfPost, int * kxPre, int * kyPre);
   int postSynapticPatchHead(int kPre,
                             int * kxPostOut, int * kyPostOut, int * kfPostOut,
                             int * dxOut, int * dyOut, int * nxpOut, int * nypOut);



   virtual int initShrinkPatches();

   virtual int shrinkPatches(int arborId);
   int shrinkPatch(int kExt, int arborId);
   bool getShrinkPatches_flag() {return shrinkPatches_flag;}

   virtual int initNormalize();
   int sumWeights(int nx, int ny, int offset, pvdata_t * dataStart, double * sum, double * sum2, pvdata_t * maxVal);
   int scaleWeights(int nx, int ny, int offset, pvdata_t * dataStart, pvdata_t sum, pvdata_t sum2, pvdata_t maxVal);
   virtual int checkNormalizeWeights(float sum, float sigma2, float maxVal);
   virtual int checkNormalizeArbor(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId);
   virtual int normalizeWeights(PVPatch ** patches, pvdata_t ** dataStart, int numPatches, int arborId);

#ifdef OBSOLETE // Marked obsolete Feb. 29, 2012.  There is no kernelIndexToPatchIndex().  There has never been a kernelIndexToPatchIndex().
   virtual int kernelIndexToPatchIndex(int kernelIndex, int * kxPatchIndex = NULL,
        int * kyPatchIndex = NULL, int * kfPatchIndex = NULL);
#endif // OBSOLETE

// patchIndexToKernelIndex() is deprecated.  Use patchIndexToDataIndex() or dataIndexToUnitCellIndex() instead
/*
   virtual int patchIndexToKernelIndex(int patchIndex, int * kxKernelIndex = NULL,
         int * kyKernelIndex = NULL, int * kfKernelIndex = NULL);
*/

   virtual int patchToDataLUT(int patchIndex);
   virtual int patchIndexToDataIndex(int patchIndex, int * kx=NULL, int * ky=NULL, int * kf=NULL);
   virtual int dataIndexToUnitCellIndex(int dataIndex, int * kx=NULL, int * ky=NULL, int * kf=NULL);

protected:
   HyPerLayer     * pre;
   HyPerLayer     * post;
   HyPerCol       * parent;
   //these were moved to private to ensure use of get/set methods and made in 3D pointers:
   //PVPatch       ** wPatches[MAX_ARBOR_LIST]; // list of weight patches, one set per neighbor
private:
   PVPatch       *** wPatches; // list of weight patches, one set per arbor
   pvdata_t      *** gSynPatchStart; //  gSynPatchStart[arborId][kExt] is a pointer to the start of the patch in the post-synaptic GSyn buffer
   pvdata_t      ** gSynPatchStartBuffer;
   size_t        ** aPostOffset; // aPostOffset[arborId][kExt] is the index of the start of a patch into an extended post-synaptic layer
   size_t         * aPostOffsetBuffer;
   int           *  delays; // delays[arborId] is the delay in timesteps (not units of dt) of the arborId'th arbor
   PVPatchStrides  postExtStrides; // sx,sy,sf for a patch mapping into an extended post-synaptic layer
   PVPatchStrides  postNonextStrides; // sx,sy,sf for a patch mapping into a non-extended post-synaptic layer
   pvdata_t      ** wDataStart; //now that data for all patches are allocated to one continuous block of memory, this pointer saves the starting address of that array
   pvdata_t      ** dwDataStart; //now that data for all patches are allocated to one continuous block of memory, this pointer saves the starting address of that array


   bool selfFlag; // indicates that connection is from a layer to itself (even though pre and post may be separately instantiated)
   bool combine_dW_with_W_flag; // indicates that dwDataStart should be set equal to wDataStart, useful for saving memory when weights are not being learned but not used
   int defaultDelay; //added to save params file defined delay...

protected:
   char * name;
   int nxp, nyp, nfp;      // size of weight dimensions
   int sxp, syp, sfp;    // stride in x,y,features

   ChannelType channel;    // which channel of the post to update (e.g. inhibit)
   int connId;             // connection id

   // PVPatch       *** dwPatches;      // list of weight patches for storing changes to weights
   int numAxonalArborLists;  // number of axonal arbors (weight patches) for presynaptic layer

   PVPatch       *** wPostPatches;  // post-synaptic linkage of weights // This is being deprecated in favor of TransposeConn
   pvdata_t      **  wPostDataStart;
   int nxpPost, nypPost, nfpPost;

   int numParams;
   //PVConnParams * params;

   float wMax;
   float wMin;

   int numProbes;
   BaseConnectionProbe ** probes; // probes used to output data
   bool ioAppend;               // controls opening of binary files
   float wPostTime;             // time of last conversion to wPostPatches
   float writeTime;             // time of next output
   float writeStep;             // output time interval

   bool writeCompressedWeights; // true=write weights with 8-bit precision;
                                // false=write weights with float precision

   int fileType;                // type ID for file written by PV::writeWeights

   Timer * update_timer;

   bool plasticityFlag;

   bool normalize_flag;
   float normalize_strength;
   bool normalize_arbors_individually;  // if true, each arbor is normalized individually, otherwise, arbors normalized together
   bool normalize_max;
   bool normalize_zero_offset;
   float normalize_cutoff;
   bool shrinkPatches_flag;

   //This object handles calculating weights.  All the initialize weights methods for all connection classes
   //are being moved into subclasses of this object.  The default root InitWeights class will create
   //2D Gaussian weights.  If weight initialization type isn't created in a way supported by Buildandrun,
   //this class will try to read the weights from a file or will do a 2D Gaussian.
   InitWeights *weightInitializer;

protected:
   inline PVPatch *** get_wPatches() {return wPatches;} // protected so derived classes can use; public methods are weights(arbor) and getWeights(patchindex,arbor)
   inline void set_wPatches(PVPatch *** patches) {wPatches=patches;}
   inline pvdata_t *** getGSynPatchStart() {return gSynPatchStart;}
   inline void setGSynPatchStart(pvdata_t *** patchstart) {gSynPatchStart = patchstart;}
   inline size_t ** getAPostOffset() {return aPostOffset;}
   inline void setAPostOffset(size_t ** postoffset) {aPostOffset = postoffset;}
   inline pvdata_t ** get_wDataStart() {return wDataStart;}
   inline void set_wDataStart(pvdata_t ** datastart) {wDataStart = datastart;}
   inline void set_wDataStart(int arborId, pvdata_t * pDataStart) {wDataStart[arborId]=pDataStart;}
   inline pvdata_t ** get_dwDataStart() {return dwDataStart;}
   inline void set_dwDataStart(pvdata_t ** datastart) {dwDataStart = datastart;}
   inline void set_dwDataStart(int arborId, pvdata_t * pIncrStart) {dwDataStart[arborId]=pIncrStart;}
   inline int * getDelays() {return delays;}
   inline void setDelays(int * delayptr) {delays = delayptr;}

   int calcUnitCellIndex(int patchIndex, int * kxUnitCellIndex=NULL, int * kyUnitCellIndex=NULL, int * kfUnitCellIndex=NULL);

   virtual int setPatchSize(const char * filename);
   virtual int setPatchStrides();
   virtual int checkPatchSize(int patchSize, int scalePre, int scalePost, char dim);
   int calcPatchSize(int n, int kex,
                     int * kl, int * offset,
                     int * nxPatch, int * nyPatch,
                     int * dx, int * dy);

   int patchSizeFromFile(const char * filename);

   int initialize_base();
   virtual int createArbors();
   void createArborsOutOfMemory();
   virtual int constructWeights(const char * filename);
#ifdef OBSOLETE // Marked obsolete Oct 1, 2011.  Made redundant by adding default value to weightInit argument of other initialize method
   int initialize(const char * name, HyPerCol * hc, HyPerLayer * pre,
         HyPerLayer * post, ChannelType channel, const char * filename);
#endif // OBSOLETE
   int initialize(const char * name, HyPerCol * hc,
                  HyPerLayer * pre, HyPerLayer * post,
                  ChannelType channel, const char * filename,
                  InitWeights *weightInit=NULL);
   virtual int initPlasticityPatches();
   virtual PVPatch *** initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches,
         const char * filename);
   virtual InitWeights * handleMissingInitWeights(PVParams * params);
//   virtual PVPatch ** createWeights(PVPatch ** patches, int nPatches, int nxPatch,
//         int nyPatch, int nfPatch, int axonId);
//   PVPatch ** createWeights(PVPatch ** patches, int axonId);
//   virtual PVPatch ** allocWeights(PVPatch ** patches, int nPatches, int nxPatch,
//         int nyPatch, int nfPatch, int axonId);
   virtual pvdata_t * createWeights(PVPatch *** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);
   pvdata_t * createWeights(PVPatch *** patches, int axonId);
   virtual pvdata_t * allocWeights(PVPatch *** patches, int nPatches, int nxPatch,
         int nyPatch, int nfPatch, int axonId);
   //PVPatch ** allocWeights(PVPatch ** patches);
   int clearWeights(pvdata_t ** dataStart, int numPatches, int nx, int ny, int nf);

   virtual int checkPVPFileHeader(Communicator * comm, const PVLayerLoc * loc, int params[], int numParams);
   virtual int checkWeightsHeader(const char * filename, int wgtParams[]);

   // virtual int deleteWeights(); // Changed to a private method.  Should not be virtual since it's called from the destructor.

   virtual int adjustAxonalArbors(int arborId);

   char * checkpointFilename();

   // following is overridden by KernelConn to set kernelPatches
   //inline void setWPatches(PVPatch ** patches, int arborId) {wPatches[arborId]=patches;}
   //virtual int setWPatches(PVPatch ** patches, int arborId) {wPatches[arborId]=patches; return 0;}
   //  int setdWPatches(PVPatch ** patches, int arborId) {dwPatches[arborId]=patches; return 0;}
   // inline void setArbor(PVAxonalArbor* arbor, int arborId) {axonalArborList[arborId]=arbor;}
   virtual int calc_dW(int axonId = 0);

   void connOutOfMemory(const char * funcname);
#ifdef PV_USE_OPENCL
   void initUseGPUFlag();
   int initializeGPU(); //this method setups up GPU stuff...
   virtual int initializeThreadBuffers(const char * kernelName);
   virtual int initializeThreadKernels(const char * kernelName);

   CLKernel * krRecvSyn;        // CL kernel for layer recvSynapticInput call
   cl_event * evRecvSynList;
   int numWait;  //number of receive synaptic runs to wait for (=numarbors)
   cl_event   evCopyDataStore;

   size_t nxl;
   size_t nyl;
   // OpenCL buffers
   //
   CLBuffer *  clGSyn;
   //CLBuffer *   clGSynSemaphors;
   //int *     gSynSemaphors; //only saving this so it can be deallocated...
   CLBuffer *  clActivity;
   CLBuffer ** clWeights;

   // ids of OpenCL arguments that change
   //
   int clArgIdOffset;
   int clArgIdWeights;
   int clArgIdDataStore;

public:
   bool gpuAccelerateFlag;
   bool ignoreGPUflag;

#endif

private:
   int clearWeights(pvdata_t * arborDataStart, int numPatches, int nx, int ny, int nf);
   int deleteWeights();

public:

   // static member functions
   //

   static PVPatch ** createPatches(int nPatches, int nx, int ny)
   {
      PVPatch ** patchpointers = (PVPatch**) malloc(nPatches*sizeof(PVPatch*));
      PVPatch * patcharray = (PVPatch*) malloc(nPatches*sizeof(PVPatch));

      PVPatch * curpatch = patcharray;
      for (int i = 0; i < nPatches; i++) {
         pvpatch_init(curpatch, nx, ny);
         patchpointers[i] = curpatch;
         curpatch++;
      }

      return patchpointers;
   }

   static int deletePatches(PVPatch ** patchpointers)
   {
      if (patchpointers != NULL && *patchpointers != NULL){
         free(*patchpointers);
         *patchpointers = NULL;
      }
      free(patchpointers);
      patchpointers = NULL;
//      for (int i = 0; i < numBundles; i++) {
//         pvpatch_inplace_delete(patches[i]);
//      }
      //free(patches);

      return 0;
   }

};

} // namespace PV

#endif /* HYPERCONN_HPP_ */
