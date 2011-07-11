/*
 * Image.hpp
 *
 *  Created on: Sep 8, 2009
 *      Author: rasmussn
 */

#ifndef IMAGE_HPP_
#define IMAGE_HPP_

#include "HyPerLayer.hpp"
#include "../columns/HyPerCol.hpp"

namespace PV {

class Image : public HyPerLayer {

protected:
   Image(const char * name, HyPerCol * hc);

public:
   Image(const char * name, HyPerCol * hc, const char * filename);
   virtual ~Image();

   virtual int initializeImage(const char * filename);

   // primary layer interface
   //
   virtual int recvSynapticInput(HyPerConn * conn, PVLayerCube * cube, int neighbor);
   virtual int updateState(float time, float dt);
   virtual int outputState(float time, bool last=false);

   // partially override implementation of LayerDataInterface interface
   //
   const pvdata_t * getLayerData()   { return data; }

   virtual int  clearImage();

   float lastUpdate()  { return lastUpdateTime; }

   virtual pvdata_t * getImageBuffer();
   virtual PVLayerLoc getImageLoc();

   virtual int tag();

   int read(const char * filename);
   int read(const char * filename, int offsetX, int offsetY);
   int write(const char * filename);

   int exchange();

   int toGrayScale();
   static unsigned char * convertToGrayScale(unsigned char * buf, int nx, int ny, int numBands);

   int  convolve(int width);
   void setTau(float t)                { tau = t; }

   int copyFromInteriorBuffer(unsigned char * buf, float fac);
   int copyToInteriorBuffer(unsigned char * buf, float fac);

protected:

#ifdef PV_USE_OPENCL
   virtual int initializeThreadBuffers();
   virtual int initializeThreadKernels();
#endif

   //int initializeImage(const char * filename);

   MPI_Datatype * mpi_datatypes;  // MPI datatypes for boundary exchange

   pvdata_t * data;       // buffer containing reduced image
   char * filename;       // path to file if a file exists

   PVLayerLoc imageLoc;   // size/location of actual image
   pvdata_t * imageData;  // buffer containing image

   int writeImages;      // controls writing of image file during outputState
   // bool useGrayScale;    // whether to convert image to grayscale
   // useGrayScale no longer used; instead setting nf=1 in params with color input images calls toGrayScale()
   int offsetX;          // offset of layer section within full image frame
   int offsetY;

   float lastPhase;
   float lastUpdateTime; // time of last image update

   float tau;
};

}

#endif /* IMAGE_HPP_ */
