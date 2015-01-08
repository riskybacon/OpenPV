#########################################
##  plotWeights.py
##  Written by Dylan Paiton, William Shainin
##  Dec 5, 2014
##
##  Input is PetaVision weight struct, generated by pvAnalysis.py
##  Produces weight plot and makes file if requested
##  Output is handle to plot or -1 if no plot is requested
##
## TODO: Multi-layer networks - multiple methods for displaying upper-layer elements
##          spike triggered average
##          regression (same as sta?)
##          Gar's deconvolution method
#########################################

#TODO: Can we plan out the imports better so they are only imported once & when needed?
#      What is proper importing protocol?
import numpy as np
import matplotlib.pyplot as plt # only need if showPlot==True OR savePlot==true
import os                       # only needed if savePlot==true
import pdb

def plotWeights(weightStruct,arborIdx=None,i_frame=0,margin=0,showPlot=False,savePlot=False,saveName=''):
    # NOTE: i_arbor and i_frame are indices for the given frame or arbor.
    #       They are not the actual arbor/frame number. This is because
    #       there may be a writeStep that is not 1.
    #
    # TODO: This would not be necessary if we knew writeStep, which we
    #       would know if we included this function in a suite that reads
    #       in parameter files.

    # weightStruct should be dims [time, numArbors, numPatches, nyp, nxp, nfp]
    weight_vals = np.array(weightStruct["values"])
    weight_time = weightStruct["time"]

    (numFrames,numArbors,numPatches,nyp,nxp,nfp) = weight_vals.shape

    if i_frame is -1:
        i_frame = numFrames-1

    if arborIdx is None:
       arborIdx = np.arange(numArbors)

    if i_frame > numFrames:
        print("Warning: i_frame > numFrames. Setting i_frame to numFrames.")
        i_frame = numFrames

    out_list = len(arborIdx) * [None] # pre-allocate list to hold weight matrices
    for i_arbor in arborIdx:
       if i_arbor > numArbors:
           print("Warning: i_arbor > numArbors. Setting i_arbor to numArbors.")
           i_arbor = numArbors

       if np.sqrt(numPatches)%1 == 0: #If numPaches has a square root
           numPatchesX = np.sqrt(numPatches)
           numPatchesY = numPatchesX
       else:
           numPatchesX = np.ceil(np.sqrt(numPatches))
           numPatchesY = numPatchesX

       patch_set = np.zeros((numPatchesX*numPatchesY,nyp+margin,nxp+margin,nfp))
       out_mat   = np.zeros((numPatchesY*(nyp+margin),numPatchesX*(nxp+margin),nfp))

       xpos = 0
       ypos = 0
       half_margin = np.floor(margin/2)

       for i_patch in range(numPatches):
           patch_tmp = weight_vals[i_frame,i_arbor,i_patch,:,:,:]
           min_patch = np.amin(patch_tmp)
           max_patch = np.amax(patch_tmp)
           # Normalize patch
           patch_tmp = (patch_tmp - min_patch) * 255 / (max_patch - min_patch + np.finfo(float).eps) # re-scaling & normalizing TODO: why? and what exactly is it doing?
           # Patches are padded with zeros - just fill in center
           patch_set[i_patch,half_margin:half_margin+nyp,half_margin:half_margin+nxp,:] = np.uint8(np.squeeze(np.transpose(patch_tmp,(1,0,2)))) # re-ordering to [x,y,f] TODO: why?

           out_mat[ypos:ypos+nyp+margin,xpos:xpos+nxp+margin,:] = patch_set[i_patch,:,:,:]

           xpos += nxp+margin

           if xpos > out_mat.shape[1]-(nxp+margin):
               ypos += nyp+margin
               xpos = 0

       out_list[i_arbor] = out_mat

       if showPlot:
           for feat in range(nfp):
               plt.figure()
               plt.imshow(out_mat[:,:,feat],cmap='Greys',interpolation='nearest')
               plt.show(block=False)
       if savePlot:
           #TODO: Should be able to pass figure title?
           for feat in range(nfp):
               if len(saveName) == 0:
                   fileName = 'plotWeightsOutput_fr'+str(i_frame).zfill(3)+'_a'+str(i_arbor).zfill(3)+'_fe'+str(feat).zfill(3)
                   fileExt  = 'png'
                   filePath = './'
                   saveName = filePath+fileName+'.'+fileExt
               else:
                   seps     = saveName.split(os.sep)
                   fileName = seps[-1]
                   filePath = saveName[0:-len(fileName)]
                   seps     = fileName.split(os.extsep)
                   fileExt  = seps[-1]
                   fileName = seps[0]
                   if not os.path.exists(filePath):
                       os.makedirs(filePath)
               plt.imsave(filePath+fileName+'_fr'+str(i_frame).zfill(3)+'_a'+str(i_arbor).zfill(3)+'_fe'+str(feat).zfill(2)+'.'+fileExt,out_mat,vmin=0,vmax=255,cmap='Greys',origin='upper')

    return out_list

#TODO:
#def plotSortedWeights(...):
#def plotWeightMovie(...):  # can receive weight file with multiple frames OR path to checkpoint folder
#def plotWeightHistograms(...):
#def plotActivationHistory(...): # activation histogram over time for each dictionary element
