/*
 * LayerLoc.h
 */

#ifndef LAYERLOC_H_
#define LAYERLOC_H_

/* The common type for data */
#define pvdata_t float

/**
 * LayerLoc describes the local location of a layer within the global space
 */
typedef struct LayerLoc_ {
   int nx, ny;             // local number of grid pts in each dimension
   int nxGlobal, nyGlobal; // total number of grid pts in the global space
   int kx0, ky0;  // origin of the local layer in global index space
   int nPad;      // size of border padding surrounding layer
   int nBands;    // number of bands (e.g., color)
} LayerLoc;

#endif /* LAYERLOC_H_ */
