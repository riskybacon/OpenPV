-- PetaVision params file for 1st level dictionary of spatiotemporal elements for detecting objects in imageNet video:
-- createded by garkenyon Dec 12, 2015

-- Load util module in PV trunk: NOTE this may need to change
--package.path = package.path .. ";" .. os.getenv("HOME") .. "/workspace/pv-core/parameterWrapper/PVModule.lua"
package.path = package.path .. ";" .. os.getenv("HOME") .. "/openpv/pv-core/parameterWrapper/PVModule.lua"
local pv = require "PVModule"

-- Global variable, for debug parsing
-- Needed by printConsole
debugParsing              = true

--HyPerLCA parameters
local VThresh               = 0.025
local VWidth                = infinity
local learningRate          = 0
local dWMax                 = 10.0
local momentumTau           = 500
local patchSizeX            = 32 --16
local patchSizeY            = 16
local tau                   = 400
local nf_Image              = 3 -- 3 for RGB, 1 for gray
local S1_numFeatures        = patchSizeX * patchSizeY * nf_Image * 2 -- (patchSize/stride)^2 Xs overcomplete (i.e. complete for orthonormal ICA basis for stride == patchSize)
local z                     = 2  -- z = 0,1,2, ... to obtain overcompleteness of 1,4,16, ...
local overcompleteness      = 2*math.pow(math.pow(2,z),2)
local overcompletenessTmp   = 4 --overcompleteness
local temporalKernelSize    = 2 -- 4 -- 
local numFrames             = 2*(temporalKernelSize * 2 - 1)

--i/o parameters
local imageListPathLeft   = "/nh/compneuro/Data/KITTI/list/image_02.txt"
local imageListPathRight  = "/nh/compneuro/Data/KITTI/list/image_03.txt"
local GroundTruthPath     = "/nh/compneuro/Data/KITTI/list/depth.txt"
local startFrame          = 0
local skipFrame           = numFrames
local offsetX             = 0
local offsetY             = 0

-- User defined variables
local plasticityFlag      = true --false
local strideX             = patchSizeX/(math.pow(2,z)*2) -- divide patchSize by 2^z to obtain dictionaries that are (2^z)^2 Xs overcomplete
local strideY             = patchSizeY/math.pow(2,z) -- 
local nxSize              = 512
local nySize              = 144
local experimentName      = "KITTI_S1X" .. overcompleteness    .. "_" .. patchSizeX .. "X" .. patchSizeY  .. "_" .. temporalKernelSize .. "X" .. numFrames    .. "frames"
local runName             = "2011_09_26_train"
local runVersion          = 5
local machinePath         = "/home/gkenyon" --"/nh/compneuro/Data"
local databasePath        = "KITTI" 
local outputPath          = machinePath .. "/" .. databasePath .. "/" .. experimentName .. "/" .. runName .. runVersion
local inputPath           = machinePath .. "/" .. databasePath .. "/" .. experimentName .. "/" .. runName .. runVersion-1
local inputPathSLP        = machinePath .. "/" .. databasePath .. "/" .. experimentName .. "/" .. runName .. runVersion-1
local numImages           = 15884
local displayPeriod       = 900 --1200
local numEpochs           = 1
local stopTime            = numImages * displayPeriod * numEpochs / numFrames
local checkpointID        = "00600000" --stopTime-- 
local checkpointID_SLP    = "00600000" --stopTime--
local writePeriod         = 100 * displayPeriod
local initialWriteTime    = writePeriod
local checkpointWriteStepInterval = writePeriod
local S1_Movie            = false
local movieVersion        = 1
if S1_Movie then
   outputPath              = outputPath .. "_S1_Movie" .. movieVersion
   inputPath               = inputPath .. "_S1_Movie" .. movieVersion - 1
   inputPathSLP            = inputPathSLP .. "_S1_Movie" .. movieVersion - 1
   displayPeriod           = 1
   writePeriod             = 1
   initialWriteTime        = numImages*(numEpochs-1)+1
   checkpointWriteStepInterval = numImages
else -- not used if run version == 1
   --inputPath               = nil -- inputPath .. runVersion-1
   --inputPathSLP            = nil --inputPath .. "S1_Movie"
end
local inf                 = 3.40282e+38
local initializeFromCheckpointFlag = false

--HyPerCol parameters
local dtAdaptFlag              = not S1_Movie
local useAdaptMethodExp1stOrder = true
local dtAdaptController        = "S1EnergyProbe"
local dtAdaptTriggerLayerName  = "FrameLeft0";
local dtScaleMax               = 0.05025 --0.0015 --0.000051   --1.0     -- minimum value for the maximum time scale, regardless of tau_eff
local dtScaleMin               = 0.05 --0.001  --0.00005  --0.01    -- default time scale to use after image flips or when something is wacky
local dtChangeMax              = 0.005   --0.1     -- determines fraction of tau_effective to which to set the time step, can be a small percentage as tau_eff can be huge
local dtChangeMin              = 0.025  --0.01    -- percentage increase in the maximum allowed time scale whenever the time scale equals the current maximum
local dtMinToleratedTimeScale  = 0.000000001

--Ground Truth parameters
local numClasses            = 32
local nxScale_GroundTruth   = 0.125
local nyScale_GroundTruth   = 0.125


-- Base table variable to store
local pvParams = {
   column = {
      groupType                           = "HyPerCol"; 
      startTime                           = 0;
      dt                                  = 1;
      dtAdaptFlag                         = dtAdaptFlag;
      useAdaptMethodExp1stOrder           = useAdaptMethodExp1stOrder;
      dtAdaptController                   = dtAdaptController;
      dtAdaptTriggerLayerName             = dtAdaptTriggerLayerName;
      dtScaleMax                          = dtScaleMax;    
      dtScaleMin                          = dtScaleMin;
      dtChangeMax                         = dtChangeMax;
      dtChangeMin                         = dtChangeMin;
      dtMinToleratedTimeScale             = dtMinToleratedTimeScale;
      stopTime                            = stopTime;
      progressInterval                    = displayPeriod;
      writeProgressToErr                  = true;
      verifyWrites                        = false;
      outputPath                          = outputPath;
      printParamsFilename                 = experimentName .. "_" .. runName .. ".params";
      randomSeed                          = 1234567890;
      nx                                  = nxSize;
      ny                                  = nySize;
      filenamesContainLayerNames          = 2; --true;
      filenamesContainConnectionNames     = 2; --true;
      initializeFromCheckpointDir         = nil; --inputPath .. "/Checkpoints/Checkpoint" .. checkpointID;
      defaultInitializeFromCheckpointFlag = initializeFromCheckpointFlag;
      checkpointWrite                     = true;
      checkpointWriteDir                  = outputPath .. "/Checkpoints";
      checkpointWriteTriggerMode          = "step";
      checkpointWriteStepInterval         = checkpointWriteStepInterval;
      deleteOlderCheckpoints              = false;
      suppressNonplasticCheckpoints       = false;
      writeTimescales                     = true;
      errorOnNotANumber                   = false;
   }
} --End of pvParams
if checkpointID then
   pvParams.column.initializeFromCheckpointDir = inputPath .. "/Checkpoints/Checkpoint" .. checkpointID;
end
if S1_Movie then
   pvParams.column.dtAdaptFlag                         = false;
   pvParams.column.dtScaleMax                          = nil;
   pvParams.column.dtScaleMin                          = nil;
   pvParams.column.dtChangeMax                         = nil;
   pvParams.column.dtChangeMin                         = nil;
   pvParams.column.dtMinToleratedTimeScale             = nil;


   if GroundTruthPath then
      for i_frame = 1, numFrames do

	 local GroundTruthMoviePath                         = inputPath .. "/" .. "GroundTruth" .. i_frame-1 .. ".pvp"
	 pv.addGroup(pvParams, "GroundTruth" .. i_frame-1,
		     {
			groupType                           = "MoviePvp";
			nxScale                             = nxScale_GroundTruth;
			nyScale                             = nyScale_GroundTruth;
			nf                                  = numClasses;
			phase                               = 0;
			mirrorBCflag                        = true;
			initializeFromCheckpointFlag        = false;
			writeStep                           = displayPeriod;
			initialWriteTime                    = displayPeriod;
			sparseLayer                         = true;
			writeSparseValues                   = false;
			updateGpu                           = false;
			dataType                            = nil;
			offsetAnchor                        = "tl";
			offsetX                             = 0;
			offsetY                             = 0;
			writeImages                         = 0;
			useImageBCflag                      = false;
			autoResizeFlag                      = false;
			inverseFlag                         = false;
			normalizeLuminanceFlag              = false;
			jitterFlag                          = 0;
			padValue                            = 0;
			inputPath                           = GroundTruthMoviePath;
			displayPeriod                       = displayPeriod;
			randomMovie                         = 0;
			readPvpFile                         = true;
			start_frame_index                   = startFrame;
			skip_frame_index                    = 0;
			writeFrameToTimestamp               = true;
			flipOnTimescaleError                = true;
			resetToStartOnLoop                  = false;
		     }
	 )
      end -- i_frame
   end -- if GroundTruthPath

   for i_delay = 1, numFrames - temporalKernelSize + 1 do
      local S1MoviePath                                   = inputPath .. "/" .. "S1.pvp"
      pv.addGroup(pvParams, "S1" .. i_delay-1,
		  {
		     groupType = "MoviePvp";
		     nxScale                             = 1.0/strideX;
		     nyScale                             = 1.0/strideY;
		     nf                                  = S1_numFeatures;
		     phase                               = 0;
		     mirrorBCflag                        = false;
		     initializeFromCheckpointFlag        = false;
		     writeStep                           = -1;
		     sparseLayer                         = true;
		     writeSparseValues                   = true;
		     updateGpu                           = false;
		     dataType                            = nil;
		     offsetAnchor                        = "tl";
		     offsetX                             = 0;
		     offsetY                             = 0;
		     writeImages                         = 0;
		     useImageBCflag                      = false;
		     autoResizeFlag                      = false;
		     inverseFlag                         = false;
		     normalizeLuminanceFlag              = false;
		     jitterFlag                          = 0;
		     padValue                            = 0;
		     inputPath                           = S1MoviePath;
		     displayPeriod                       = displayPeriod;
		     randomMovie                         = 0;
		     readPvpFile                         = true;
		     start_frame_index                   = startFrame;
		     skip_frame_index                    = 0;
		     writeFrameToTimestamp               = true;
		     flipOnTimescaleError                = true;
		     resetToStartOnLoop                  = false;
		  }
      )
   end -- i_delay

else -- not S1_Movie
   
   -- pv.addGroup(pvParams, "ConstantS1",
   -- 	       {
   -- 		  groupType = "ConstantLayer";
   -- 		  nxScale                             = 1.0/strideX;
   -- 		  nyScale                             = 1.0/strideY;
   -- 		  nf                                  = S1_numFeatures;
   -- 		  phase                               = 0;
   -- 		  mirrorBCflag                        = false;
   -- 		  valueBC                             = 0;
   -- 		  initializeFromCheckpointFlag        = initializeFromCheckpointFlag;
   -- 		  InitVType                           = "ConstantV";
   -- 		  valueV                              = VThresh;
   -- 		  writeStep                           = -1;
   -- 		  sparseLayer                         = false;
   -- 		  updateGpu                           = false;
   -- 		  dataType                            = nil;
   -- 		  VThresh                             = -inf;
   -- 		  AMin                                = -inf;
   -- 		  AMax                                = inf;
   -- 		  AShift                              = 0;
   -- 		  VWidth                              = 0;
   -- 		  clearGSynInterval                   = 0;
   -- 	       }
   -- )

   --for i_frame = 1, numFrames+temporalKernelSize-1 do
   for i_frame = 1, numFrames do
      
      pv.addGroup(pvParams,
		  "FrameLeft" .. i_frame-1, 
		  {
		     groupType                           = "Movie";
		     nxScale                             = 1;
		     nyScale                             = 1;
		     nf                                  = nf_Image;
		     phase                               = 0;
		     mirrorBCflag                        = true;
		     initializeFromCheckpointFlag        = false;
		     writeStep                           = writePeriod;
		     initialWriteTime                    = initialWriteTime;
		     sparseLayer                         = false;
		     writeSparseValues                   = false;
		     updateGpu                           = false;
		     dataType                            = nil;
		     offsetAnchor                        = "tl";
		     offsetX                             = offsetX;
		     offsetY                             = offsetY;
		     writeImages                         = 0;
		     useImageBCflag                      = false;
		     autoResizeFlag                      = true;
		     inverseFlag                         = false;
		     normalizeLuminanceFlag              = true;
		     normalizeStdDev                     = true;
		     jitterFlag                          = 0;
		     padValue                            = 0;
		     inputPath                           = imageListPathLeft;
		     displayPeriod                       = displayPeriod;
		     echoFramePathnameFlag               = true;
		     start_frame_index                   = startFrame + i_frame-1;
		     skip_frame_index                    = skipFrame;
		     writeFrameToTimestamp               = true;
		     flipOnTimescaleError                = true;
		     resetToStartOnLoop                  = false;
		  }
      )
      pv.addGroup(pvParams,
		  "FrameRight" .. i_frame-1, pvParams["FrameLeft" .. i_frame-1],
		  {
		     inputPath                           = imageListPathRight;
		  }
      )
      
   end -- i_frame

   for i_frame = 1, numFrames do  

      pv.addGroup(pvParams,
		  "FrameLeft" .. i_frame-1 .. "ReconS1Error",
		  {
		     groupType                           = "ANNLayer";
		     nxScale                             = 1;
		     nyScale                             = 1;	
		     nf                                  = nf_Image;
		     phase                               = 2;
		     mirrorBCflag                        = false;
		     valueBC                             = 0;
		     initializeFromCheckpointFlag        = initializeFromCheckpointFlag;
		     InitVType                           = "ZeroV";
		     triggerLayerName                    = NULL;
		     writeStep                           = writePeriod;
		     initialWriteTime                    = initialWriteTime;
		     sparseLayer                         = false;
		     updateGpu                           = false;
		     dataType                            = nil;
		     VThresh                             = -inf;
		     AMin                                = -inf;
		     AMax                                = inf;
		     AShift                              = 0;
		     VWidth                              = 0;
		     clearGSynInterval                   = 0;
		     errScale                            = 1;
		  }
      )
      pv.addGroup(pvParams,
		  "FrameRight" .. i_frame-1 .. "ReconS1Error",
		  pvParams["FrameLeft" .. i_frame-1 .. "ReconS1Error"])
      
   end -- i_frame
   
   for i_delay = 1, numFrames - temporalKernelSize + 1 do
   --for i_delay = 1, numFrames do
      pv.addGroup(pvParams,
		  "S1_" .. i_delay-1,
		  {
		     groupType = "HyPerLCALayer";
		     nxScale                             = 1.0/strideX;
		     nyScale                             = 1.0/strideY;
		     nf                                  = S1_numFeatures;
		     phase                               = 2;
		     mirrorBCflag                        = false;
		     valueBC                             = 0;
		     initializeFromCheckpointFlag        = initializeFromCheckpointFlag;
		     --InitVType                           = "InitVFromFile";
		     --Vfilename                           = inputPath .. "/Checkpoints/Checkpoint" .. checkpointID .. "/S1_V.pvp";
		     InitVType                           = "UniformRandomV";
		     minV                                = -VThresh;
		     maxV                                = 2*VThresh;
		     --InitVType                           = "ConstantV";
		     --valueV                              = VThresh;
		     --triggerLayerName                    = "FrameLeft" .. i_frame-1;
		     --triggerBehavior                     = "resetStateOnTrigger";
		     --triggerResetLayerName               = "ConstantS1";
		     --triggerOffset                       = 0;
		     writeStep                           = displayPeriod;
		     initialWriteTime                    = displayPeriod;
		     sparseLayer                         = true;
		     writeSparseValues                   = true;
		     updateGpu                           = true;
		     dataType                            = nil;
		     VThresh                             = VThresh;
		     AMin                                = 0;
		     AMax                                = inf;
		     AShift                              = 0;
		     VWidth                              = 100;
		     clearGSynInterval                   = 0;
		     numChannels                         = 1;
		     timeConstantTau                     = tau;
		     numWindowX                          = 1;
		     numWindowY                          = 1;
		     selfInteract                        = true;
		  }
      )
      if checkpointID then
	 pvParams["S1_" .. i_delay-1].initializeFromCheckpointFlag = true;
      end   
   end -- i_delay

   for i_frame = 1, numFrames do  
   --for i_frame = 1, numFrames+temporalKernelSize-1 do  

      pv.addGroup(pvParams, "FrameLeft" .. i_frame-1 .. "ReconS1",
		  {
		     groupType = "ANNLayer";
		     nxScale                             = 1;
		     nyScale                             = 1;
		     nf                                  = nf_Image;
		     phase                               = 4;
		     mirrorBCflag                        = false;
		     valueBC                             = 0;
		     initializeFromCheckpointFlag        = initializeFromCheckpointFlag;
		     InitVType                           = "ZeroV";
		     triggerLayerName                    = NULL;
		     writeStep                           = writePeriod;
		     initialWriteTime                    = writePeriod;
		     sparseLayer                         = false;
		     updateGpu                           = false;
		     dataType                            = nil;
		     VThresh                             = -inf;
		     AMin                                = -inf;
		     AMax                                = inf;
		     AShift                              = 0;
		     VWidth                              = 0;
		     clearGSynInterval                   = 0;
		  }
      )
      if i_frame > numFrames then
	 pvParams["FrameLeft" .. i_frame-1 .. "ReconS1"].triggerLayerName = "FrameLeft" .. i_frame-1;
	 pvParams["FrameLeft" .. i_frame-1 .. "ReconS1"].triggerOffset     = 1;
      end
      pv.addGroup(pvParams, "FrameRight" .. i_frame-1 .. "ReconS1",
		  pvParams["FrameLeft" .. i_frame-1 .. "ReconS1"])

      for i_delay = 1, numFrames - temporalKernelSize + 1 do
      --for i_delay = 1, numFrames do

	 local delta_frame = i_frame - i_delay
	 if (delta_frame >= 0 and delta_frame < temporalKernelSize) then
	    
	    pv.addGroup(pvParams, "FrameLeft" .. i_frame-1 .. "ReconS1_" .. i_delay-1,
			pvParams["FrameLeft" .. i_frame-1 .. "ReconS1"], 
			{
			   phase                               = 3;
			}
	    )
	    pv.addGroup(pvParams,
			"FrameRight" .. i_frame-1 .. "ReconS1_" .. i_delay-1,
			pvParams["FrameLeft" .. i_frame-1 .. "ReconS1_" .. i_delay-1])

	 end -- delta_frame >= 0
      end -- i_delay
   end -- i_frame

end -- S1_Movie

-- Ground Truth 

if not S1_Movie then
   if GroundTruthPath then
      for i_frame = 1, numFrames do
	 
	 pv.addGroup(pvParams, "GroundTruthPixels" .. i_frame-1,
		     {
		     groupType = "Movie";
		     nxScale                             = 1;
		     nyScale                             = 1;
		     nf                                  = numClasses;
		     phase                               = 0;
		     mirrorBCflag                        = false;
		     initializeFromCheckpointFlag        = false;
		     writeStep                           = writePeriod;
		     initialWriteTime                    = initialWriteTime;
		     sparseLayer                         = false;
		     writeSparseValues                   = false;
		     updateGpu                           = false;
		     dataType                            = nil;
		     offsetAnchor                        = "tl";
		     offsetX                             = offsetX;
		     offsetY                             = offsetY;
		     writeImages                         = 0;
		     useImageBCflag                      = false;
		     autoResizeFlag                      = true;
		     inverseFlag                         = false;
		     normalizeLuminanceFlag              = true;
		     normalizeStdDev                     = true;
		     jitterFlag                          = 0;
		     padValue                            = 0;
		     inputPath                           = GroundTruthPath;
		     displayPeriod                       = displayPeriod;
		     echoFramePathnameFlag               = true;
		     start_frame_index                   = startFrame + i_frame-1;
		     skip_frame_index                    = skipFrame;
		     writeFrameToTimestamp               = true;
		     flipOnTimescaleError                = true;
		     resetToStartOnLoop                  = false;
		     }
	 )
	 
	 pv.addGroup(pvParams, "GroundTruth" .. i_frame-1,
		     pvParams["FrameLeft" .. i_frame-1 .. "ReconS1"],
		     {
			nxScale                             = nxScale_GroundTruth;
			nyScale                             = nyScale_GroundTruth;
			nf                                  = numClasses;
			phase                               = 1;
			writeStep                           = displayPeriod;
			initialWriteTime                    = displayPeriod;
			sparseLayer                         = true;
		     }
	 )
	 pvParams["GroundTruth" .. i_frame-1].triggerLayerName  = "GroundTruthPixels" .. i_frame-1;
	 pvParams["GroundTruth" .. i_frame-1].triggerBehavior   = "updateOnlyOnTrigger";
	 pvParams["GroundTruth" .. i_frame-1].triggerOffset     = 0;
	 pvParams["GroundTruth" .. i_frame-1].writeSparseValues = false;
	 
      end -- i_frame
      
   end -- GroundTruthPath
end -- not S1_Movie

if GroundTruthPath then
   
   for i_frame = 1, numFrames do  
      
      pv.addGroup(pvParams, "GroundTruth" .. i_frame-1 .. "ReconS1Error",
		  {
		     groupType                           = "MaskLayer";
		     nxScale                             = nxScale_GroundTruth;
		     nyScale                             = nyScale_GroundTruth;
		     nf                                  = numClasses;
		     phase                               = 11;
		     mirrorBCflag                        = false;
		     valueBC                             = 0;
		     initializeFromCheckpointFlag        = false;
		     InitVType                           = "ZeroV";
		     triggerLayerName                    = "GroundTruthPixels" .. i_frame-1;
		     triggerBehavior                     = "updateOnlyOnTrigger";
		     triggerOffset                       = 1;
		     writeStep                           = displayPeriod;
		     initialWriteTime                    = displayPeriod;
		     sparseLayer                         = false;
		     updateGpu                           = false;
		     dataType                            = nil;
		     VThresh                             = -inf;
		     AMin                                = -inf;
		     AMax                                = inf;
		     AShift                              = 0;
		     VWidth                              = 0;
		     clearGSynInterval                   = 0;
		     maskMethod                          = "layer";
		     maskLayerName                       = "GroundTruth" .. i_frame-1
		  }
      )
      if S1_Movie then
	 pvParams["GroundTruth" .. i_frame-1 .. "ReconS1Error"].triggerLayerName  =
	    "GroundTruth" .. i_frame-1;
	 pvParams["GroundTruth" .. i_frame-1 .. "ReconS1Error"].triggerOffset  = 0;
      end
            
      pv.addGroup(pvParams,
		  "GroundTruth" .. i_frame-1 .. "ReconS1",
		  {
		     groupType = "ANNLayer";
		     nxScale                             = nxScale_GroundTruth;
		     nyScale                             = nyScale_GroundTruth;
		     nf                                  = numClasses;
		     phase                               = 10;
		     mirrorBCflag                        = false;
		     valueBC                             = 0;
		     initializeFromCheckpointFlag        = false;
		     InitVType                           = "ZeroV";
		     triggerLayerName                    = "GroundTruthPixels" .. i_frame-1;
		     triggerBehavior                     = "updateOnlyOnTrigger";
		     triggerOffset                       = 1;
		     writeStep                           = displayPeriod;
		     initialWriteTime                    = displayPeriod;
		     sparseLayer                         = false;
		     updateGpu                           = false;
		     dataType                            = nil;
		     VThresh                             = -inf;
		     AMin                                = -inf;
		     AMax                                = inf;
		     AShift                              = 0;
		     VWidth                              = 0;
		     clearGSynInterval                   = 0;
		  }
      )
      if S1_Movie then
	 pvParams["GroundTruth" .. i_frame-1 .. "ReconS1"].triggerLayerName  =
	    "GroundTruth" .. i_frame-1;
	 pvParams["GroundTruth" .. i_frame-1 .. "ReconS1"].triggerOffset  = 0;
      end
            
      for i_delay = 1, numFrames - temporalKernelSize + 1 do
	 
	 local delta_frame = i_frame - i_delay
	 if (delta_frame >= 0 and delta_frame < temporalKernelSize) then
	    	    
	    pv.addGroup(pvParams, "GroundTruth" .. i_frame-1 .. "ReconS1_" .. i_delay-1,
			pvParams[ "GroundTruth" .. i_frame-1 .. "ReconS1"],
			{
			   phase                               = 9;
			}
	    )
	    
	 end -- delta_frame >= 0
      end -- i_delay
   end -- i_frame
   

   pv.addGroup(pvParams, "BiasS1",
	       {
		  groupType = "ConstantLayer";
		  nxScale                             = nxScale_GroundTruth;
		  nyScale                             = nyScale_GroundTruth;
		  nf                                  = 1;
		  phase                               = 0;
		  mirrorBCflag                        = false;
		  valueBC                             = 0;
		  initializeFromCheckpointFlag        = false;
		  InitVType                           = "ConstantV";
		  valueV                              = 1;
		  writeStep                           = -1;
		  sparseLayer                         = false;
		  updateGpu                           = false;
		  dataType                            = nil;
		  VThresh                             = -inf;
		  AMin                                = -inf;
		  AMax                                = inf;
		  AShift                              = 0;
		  VWidth                              = 0;
		  clearGSynInterval                   = 0;
	       }
   )
   
   for i_delay = 1, numFrames - temporalKernelSize + 1 do
      pv.addGroup(pvParams,
		  "S1_" .. i_delay-1 .. "MaxPooled",
		  {
		     groupType = "ANNLayer";
		     nxScale                             = nxScale_GroundTruth;
		     nyScale                             = nyScale_GroundTruth;
		     nf                                  = S1_numFeatures;
		     phase                               = 8;
		     mirrorBCflag                        = false;
		     valueBC                             = 0;
		     initializeFromCheckpointFlag        = false;
		     InitVType                           = "ZeroV";
		     triggerLayerName                    = "GroundTruthPixels" .. 0;
		     triggerBehavior                     = "updateOnlyOnTrigger";
		     triggerOffset                       = 1;
		     writeStep                           = -1;
		     sparseLayer                         = false;
		     updateGpu                           = false;
		     dataType                            = nil;
		     VThresh                             = -inf;
		     AMin                                = -inf;
		     AMax                                = inf;
		     AShift                              = 0;
		     VWidth                              = 0;
		     clearGSynInterval                   = 0;
		  }
      )
      if S1_Movie then
	 pvParams["S1_" .. i_delay-1 .. "MaxPooled"].triggerLayerName  = "GroundTruth" .. 0;
	 pvParams["S1_" .. i_delay-1 .. "MaxPooled"].triggerOffset     = 0;
      end
      
   end -- i_delay
   
end -- GroundTruthPath



--connections

if not S1_Movie then
   for i_frame = 1, numFrames do  

      pv.addGroup(pvParams,
		  "FrameLeft" .. i_frame-1 .. "To" .. "FrameLeft" .. i_frame-1 .. "ReconS1Error",
		  {
		     groupType                           = "HyPerConn";
		     preLayerName                        = "FrameLeft" .. i_frame-1;
		     postLayerName                       = "FrameLeft" .. i_frame-1 .. "ReconS1Error";
		     channelCode                         = 0;
		     delay                               = {0.000000};
		     numAxonalArbors                     = 1;
		     plasticityFlag                      = false;
		     convertRateToSpikeCount             = false;
		     receiveGpu                          = false;
		     sharedWeights                       = true;
		     weightInitType                      = "OneToOneWeights";
		     initWeightsFile                     = nil;
		     weightInit                          = math.sqrt((1/patchSizeX)*(1/patchSizeY)*(1/nf_Image)); --
		     initializeFromCheckpointFlag        = false;
		     updateGSynFromPostPerspective       = false;
		     pvpatchAccumulateType               = "convolve";
		     writeStep                           = -1;
		     writeCompressedCheckpoints          = false;
		     selfFlag                            = false;
		     nxp                                 = 1;
		     nyp                                 = 1;
		     nfp                                 = 3;
		     shrinkPatches                       = false;
		     normalizeMethod                     = "none";
		  }
      )
      pv.addGroup(pvParams,
		  "FrameRight" .. i_frame-1 .. "To" .. "FrameRight" .. i_frame-1 .. "ReconS1Error",
		  pvParams["FrameLeft" .. i_frame-1 .. "To" .. "FrameLeft" .. i_frame-1 .. "ReconS1Error"],
		  {
		     preLayerName                        = "FrameRight" .. i_frame-1;
		     postLayerName                       = "FrameRight" .. i_frame-1 .. "ReconS1Error";
		  }
      )
      
      pv.addGroup(pvParams,
		  "FrameLeft" .. i_frame-1 .. "ReconS1" .. "To" .. "FrameLeft" .. i_frame-1 .. "ReconS1Error",
		  {
		     groupType                           = "IdentConn";
		     preLayerName                        = "FrameLeft" .. i_frame-1 .. "ReconS1";
		     postLayerName                       = "FrameLeft" .. i_frame-1 .. "ReconS1Error";
		     channelCode                         = 1;
		     delay                               = {0.000000};
		     initWeightsFile                     = nil;
		     writeStep                           = -1;
		  }
      )
      pv.addGroup(pvParams,
		  "FrameRight" .. i_frame-1 .. "ReconS1" .. "To" .. "FrameRight" .. i_frame-1 .. "ReconS1Error",
		  {
		     groupType                           = "IdentConn";
		     preLayerName                        = "FrameRight" .. i_frame-1 .. "ReconS1";
		     postLayerName                       = "FrameRight" .. i_frame-1 .. "ReconS1Error";
		     channelCode                         = 1;
		     delay                               = {0.000000};
		     initWeightsFile                     = nil;
		     writeStep                           = -1;
		  }
      )
      
      for i_delay = 1, numFrames - temporalKernelSize + 1 do
	 --for i_delay = 1, numFrames do
	 
	 local delta_frame = i_frame - i_delay
	 if (delta_frame >= 0 and delta_frame < temporalKernelSize) then
	    
	    pv.addGroup(pvParams,
			"FrameLeft" .. i_frame-1 .. "Recon" .. "S1_" .. i_delay-1 .. "To" .. "FrameLeft" .. i_frame-1 .. "ReconS1",
			{
			   groupType                           = "IdentConn";
			   preLayerName                        = "FrameLeft" .. i_frame-1 .. "ReconS1_" .. i_delay-1;
			   postLayerName                       = "FrameLeft" .. i_frame-1 .. "ReconS1";
			   channelCode                         = 0;
			   delay                               = {0.000000};
			   initWeightsFile                     = nil;
			   writeStep                           = -1;
			}
	    )
	    pv.addGroup(pvParams,
			"FrameRight" .. i_frame-1 .. "Recon" .. "S1_" .. i_delay-1 .. "To" .. "FrameRight" .. i_frame-1 .. "ReconS1",
			pvParams["FrameLeft" .. i_frame-1 .. "Recon" .. "S1_" .. i_delay-1 .. "To" .. "FrameLeft" .. i_frame-1 .. "ReconS1"],
			{
			   preLayerName                        = "FrameRight" .. i_frame-1 .. "ReconS1_" .. i_delay-1;
			   postLayerName                       = "FrameRight" .. i_frame-1 .. "ReconS1";
			}
	    )

	    if i_delay == 1 then -- the first delay layer stores the original connections

	       pv.addGroup(pvParams,
			   "S1_" .. 0 .. "To" .. "FrameLeft" .. delta_frame .. "ReconS1Error",
			   {
			      groupType                           = "MomentumConn";
			      preLayerName                        = "S1_" .. 0;
			      postLayerName                       = "FrameLeft" .. delta_frame .. "ReconS1Error";
			      channelCode                         = -1;
			      delay                               = {0.000000};
			      numAxonalArbors                     = 1;
			      plasticityFlag                      = plasticityFlag;
			      convertRateToSpikeCount             = false;
			      receiveGpu                          = false;
			      sharedWeights                       = true;
			      weightInitType                      = "UniformRandomWeight";
			      initWeightsFile                     = nil;
			      wMinInit                            = -1;
			      wMaxInit                            = 1;
			      sparseFraction                      = 0.9;
			      initializeFromCheckpointFlag        = false;
			      triggerLayerName                    = "FrameLeft" .. delta_frame;
			      triggerOffset                       = 1;
			      updateGSynFromPostPerspective       = true;
			      pvpatchAccumulateType               = "convolve";
			      writeStep                           = -1;
			      writeCompressedCheckpoints          = false;
			      selfFlag                            = false;
			      combine_dW_with_W_flag              = false;
			      nxp                                 = patchSizeX;
			      nyp                                 = patchSizeY;
			      shrinkPatches                       = false;
			      normalizeMethod                     = "normalizeL2";
			      strength                            = 1;
			      normalizeArborsIndividually         = false;
			      normalizeOnInitialize               = true;
			      normalizeOnWeightUpdate             = true;
			      rMinX                               = 0;
			      rMinY                               = 0;
			      nonnegativeConstraintFlag           = false;
			      normalize_cutoff                    = 0;
			      normalizeFromPostPerspective        = false;
			      minL2NormTolerated                  = 0;
			      dWMax                               = dWMax;
			      normalizeDw                         = true;
			      keepKernelsSynchronized             = true;
			      useMask                             = false;
			      momentumTau                         = momentumTau;
			      momentumMethod                      = "viscosity";
			      momentumDecay                       = 0;

			   }
	       )
	       pv.addGroup(pvParams,
			   "S1_" .. 0 .. "To" .. "FrameRight" .. delta_frame .. "ReconS1Error",
			   pvParams["S1_" .. 0 .. "To" .. "FrameLeft" .. delta_frame .. "ReconS1Error"],
			   {
			      postLayerName                       = "FrameRight" .. delta_frame .. "ReconS1Error";
			   }
	       )
	       pvParams["S1_" .. 0 .. "To" .. "FrameRight" .. delta_frame .. "ReconS1Error"].normalizeMethod
		  = "normalizeGroup";
	       pvParams["S1_" .. 0 .. "To" .. "FrameRight" .. delta_frame .. "ReconS1Error"].normalizeGroupName                 
		  = "S1_" .. 0 .. "To" .. "FrameLeft" .. delta_frame .. "ReconS1Error";
	       
	       if not plasticityFlag then
		  pvParams["S1_" .. 0 .. "To" .. "FrameLeft"  .. delta_frame .. "ReconS1Error"].triggerLayerName    = NULL;
		  pvParams["S1_" .. 0 .. "To" .. "FrameLeft"  .. delta_frame .. "ReconS1Error"].triggerOffset       = nil;
		  pvParams["S1_" .. 0 .. "To" .. "FrameLeft"  .. delta_frame .. "ReconS1Error"].triggerBehavior      = nil;
		  pvParams["S1_" .. 0 .. "To" .. "FrameRight" .. delta_frame .. "ReconS1Error"].triggerLayerName    = NULL;
		  pvParams["S1_" .. 0 .. "To" .. "FrameRight" .. delta_frame .. "ReconS1Error"].triggerOffset       = nil;
		  pvParams["S1_" .. 0 .. "To" .. "FrameRight" .. delta_frame .. "ReconS1Error"].triggerBehavior      = nil;
	       end
	       if checkpointID then
		  pvParams["S1_" .. 0 .. "To" .. "FrameLeft"  .. delta_frame .. "ReconS1Error"].weightInitType      = "FileWeight";
		  pvParams["S1_" .. 0 .. "To" .. "FrameLeft"  .. delta_frame .. "ReconS1Error"].initWeightsFile
		     = inputPath .. "/Checkpoints/Checkpoint" .. checkpointID .. "/" .. "S1_" .. 0 .. "ToFrameLeft" .. delta_frame .. "ReconS1Error_W.pvp";
		  pvParams["S1_" .. 0 .. "To" .. "FrameLeft"  .. delta_frame .. "ReconS1Error"].useListOfArborFiles = false;
		  pvParams["S1_" .. 0 .. "To" .. "FrameLeft"  .. delta_frame .. "ReconS1Error"].combineWeightFiles  = false;    
		  pvParams["S1_" .. 0 .. "To" .. "FrameRight" .. delta_frame .. "ReconS1Error"].weightInitType      = "FileWeight";
		  pvParams["S1_" .. 0 .. "To" .. "FrameRight" .. delta_frame .. "ReconS1Error"].initWeightsFile
		     = inputPath .. "/Checkpoints/Checkpoint" .. checkpointID .. "/" .. "S1_" .. 0 .. "ToFrameRight" .. delta_frame .. "ReconS1Error_W.pvp";
		  pvParams["S1_" .. 0 .. "To" .. "FrameRight" .. delta_frame .. "ReconS1Error"].useListOfArborFiles = false;
		  pvParams["S1_" .. 0 .. "To" .. "FrameRight" .. delta_frame .. "ReconS1Error"].combineWeightFiles  = false;    
	       end -- checkpointID	       
	       
	    else -- use a plasticCloneConn
	       
	       pv.addGroup(pvParams,
			   "S1_" .. i_delay-1 .. "To" .. "FrameLeft" .. i_frame-1 .. "ReconS1Error",

			   {
			      groupType                           = "PlasticCloneConn";
			      preLayerName                        = "S1_" .. i_delay-1;
			      postLayerName                       = "FrameLeft" .. i_frame-1 .. "ReconS1Error";
			      channelCode                         = -1;
			      delay                               = {0.000000};
			      selfFlag                            = false;
			      preActivityIsNotRate                = false;
			      updateGSynFromPostPerspective       = false;
			      pvpatchAccumulateType               = "convolve";
			      originalConnName                    = "S1_" .. 0 .. "To" .. "FrameLeft" .. delta_frame .. "ReconS1Error";
			   }
	       )
	       pv.addGroup(pvParams,
			   "S1_" .. i_delay-1 .. "To" .. "FrameRight" .. i_frame-1 .. "ReconS1Error",
			   pvParams["S1_" .. i_delay-1 .. "To" .. "FrameLeft" .. i_frame-1 .. "ReconS1Error"],

			   {
			      postLayerName                       = "FrameRight" .. i_frame-1 .. "ReconS1Error";
			      originalConnName                    = "S1_" .. 0 .. "To" .. "FrameRight" .. delta_frame .. "ReconS1Error";
			   }
	       )

	    end -- i_delay == 1

	    pv.addGroup(pvParams,
			"FrameLeft" .. i_frame-1 .. "ReconS1Error" .. "To" .. "S1_" .. i_delay-1,
			{
			   groupType                           = "TransposeConn";
			   preLayerName                        = "FrameLeft" .. i_frame-1 .. "ReconS1Error";
			   postLayerName                       = "S1_" .. i_delay-1;
			   channelCode                         = 0;
			   delay                               = {0.000000};
			   convertRateToSpikeCount             = false;
			   receiveGpu                          = true;
			   updateGSynFromPostPerspective       = true;
			   pvpatchAccumulateType               = "convolve";
			   writeStep                           = -1;
			   writeCompressedCheckpoints          = false;
			   selfFlag                            = false;
			   gpuGroupIdx                         = delta_frame;
			   originalConnName                    = "S1_" .. 0 .. "To" .. "FrameLeft" .. delta_frame .. "ReconS1Error";
			}
	    )
	    pv.addGroup(pvParams,
			"FrameRight" .. i_frame-1 .. "ReconS1Error" .. "To" .. "S1_" .. i_delay-1,
			pvParams["FrameLeft" .. i_frame-1 .. "ReconS1Error" .. "To" .. "S1_" .. i_delay-1],
			{
			   preLayerName                        = "FrameRight" .. i_frame-1 .. "ReconS1Error";
			   originalConnName                    = "S1_" .. 0 .. "To" .. "FrameRight" .. delta_frame .. "ReconS1Error";
			}
	    )
	    
	 end -- delta_frame >= 0
      end -- i_delay
   end -- i_frame
   
   --for i_frame = 1, numFrames+temporalKernelSize-1 do  
   for i_frame = 1, numFrames do  
      for i_delay = 1, numFrames - temporalKernelSize + 1 do
      --for i_delay = 1, numFrames do

	 local delta_frame = i_frame - i_delay
	 if (delta_frame >= 0 and delta_frame < temporalKernelSize) then
	    
	    pv.addGroup(pvParams,
			"S1_" .. i_delay-1 .. "To" .. "FrameLeft" .. i_frame-1 .. "ReconS1_" .. i_delay-1,
			{
			   groupType                           = "CloneConn";
			   preLayerName                        = "S1_" .. i_delay-1;
			   postLayerName                       = "FrameLeft" .. i_frame-1 .. "ReconS1_" .. i_delay-1;
			   channelCode                         = 0;
			   delay                               = {0.000000};
			   convertRateToSpikeCount             = false;
			   receiveGpu                          = false;
			   updateGSynFromPostPerspective       = false;
			   pvpatchAccumulateType               = "convolve";
			   writeStep                           = -1;
			   writeCompressedCheckpoints          = false;
			   selfFlag                            = false;
			   originalConnName                    = "S1_" .. 0 .. "ToFrameLeft" .. delta_frame .. "ReconS1Error";
			}
	    )
	    pv.addGroup(pvParams,
			"S1_" .. i_delay-1 .. "To" .. "FrameRight" .. i_frame-1 .. "ReconS1_" .. i_delay-1,
			pvParams["S1_" .. i_delay-1 .. "To" .. "FrameLeft" .. i_frame-1 .. "ReconS1_" .. i_delay-1],
			{
			   postLayerName                       = "FrameRight" .. i_frame-1 .. "ReconS1_" .. i_delay-1;
			   originalConnName                    = "S1_" .. 0 .. "ToFrameRight" .. delta_frame .. "ReconS1Error";
			}
	    )
	    
	 end -- delta_frame >= 0
      end -- i_delay
   end -- i_frame
   

   if GroundTruthPath then
      
      for i_frame = 1, numFrames do  
	 pv.addGroup(pvParams,
		     "GroundTruthPixels" .. i_frame-1 .. "ToGroundTruth" .. i_frame-1,
		     {
			groupType                           = "PoolingConn";
			preLayerName                        = "GroundTruthPixels" .. i_frame-1;
			postLayerName                       = "GroundTruth" .. i_frame-1;
			channelCode                         = 0;
			delay                               = {0.000000};
			numAxonalArbors                     = 1;
			convertRateToSpikeCount             = false;
			receiveGpu                          = false;
			sharedWeights                       = true;
			initializeFromCheckpointFlag        = false;
			updateGSynFromPostPerspective       = false;
			pvpatchAccumulateType               = "maxpooling";
			writeStep                           = -1;
			writeCompressedCheckpoints          = false;
			selfFlag                            = false;
			nxp                                 = 1;
			nyp                                 = 1;
			shrinkPatches                       = false;
			needPostIndexLayer                  = false;
		     }
	 )
      end -- i_frame
   end -- GroundTruthPath

end -- S1_Movie

-- Ground Truth connections
if GroundTruthPath then

   for i_frame = 1, numFrames do  

      pv.addGroup(pvParams,
		  "GroundTruth" .. i_frame-1 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error",
		  {
		     groupType                           = "IdentConn";
		     preLayerName                        = "GroundTruth" .. i_frame-1;
		     postLayerName                       = "GroundTruth" .. i_frame-1 .. "ReconS1Error";
		     channelCode                         = 0;
		     delay                               = {0.000000};
		     initWeightsFile                     = nil;
		     writeStep                           = -1;
		  }
      )


      pv.addGroup(pvParams,
		  "GroundTruth" .. i_frame-1 .. "ReconS1" .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error",
		  {
		     groupType                           = "IdentConn";
		     preLayerName                        = "GroundTruth" .. i_frame-1 .. "ReconS1";
		     postLayerName                       = "GroundTruth" .. i_frame-1 .. "ReconS1Error";
		     channelCode                         = 1;
		     delay                               = {0.000000};
		     initWeightsFile                     = nil;
		     writeStep                           = -1;
		  }
      )


      for i_delay = 1, numFrames - temporalKernelSize + 1 do

	 local delta_frame = i_frame - i_delay
	 if (delta_frame >= 0 and delta_frame < temporalKernelSize) then

	    
	    pv.addGroup(pvParams,
			"GroundTruth" .. i_frame-1 .. "ReconS1_" .. i_delay-1 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1",
			{
			   groupType                           = "IdentConn";
			   preLayerName                        = "GroundTruth" .. i_frame-1 .. "ReconS1_" .. i_delay-1;
			   postLayerName                       = "GroundTruth" .. i_frame-1 .. "ReconS1";
			   channelCode                         = 0;
			   delay                               = {0.000000};
			   initWeightsFile                     = nil;
			   writeStep                           = -1;
			}
	    )
	    
	    if i_delay == 1 then -- the first delay layer stores the original connections
	       
	       pv.addGroup(pvParams,
			   "S1_" .. 0  .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error",
			   {
			      groupType                           = "HyPerConn";
			      preLayerName                        = "S1_" .. 0;
			      postLayerName                       = "GroundTruth" .. i_frame-1 .. "ReconS1Error";
			      channelCode                         = -1;
			      delay                               = {0.000000};
			      numAxonalArbors                     = 1;
			      plasticityFlag                      = plasticityFlag;
			      convertRateToSpikeCount             = false;
			      receiveGpu                          = false;
			      sharedWeights                       = true;
			      weightInitType                      = "UniformRandomWeight";
			      initWeightsFile                     = nil;
			      wMinInit                            = -0;
			      wMaxInit                            = 0;
			      sparseFraction                      = 0;
			      initializeFromCheckpointFlag        = false;
			      triggerLayerName                    = "FrameLeft" .. i_frame-1;
			      triggerBehavior                     = "updateOnlyOnTrigger";
			      triggerOffset                       = 1;
			      updateGSynFromPostPerspective       = false;
			      pvpatchAccumulateType               = "convolve";
			      writeStep                           = -1;
			      writeCompressedCheckpoints          = false;
			      selfFlag                            = false;
			      combine_dW_with_W_flag              = false;
			      nxp                                 = 1;
			      nyp                                 = 1;
			      shrinkPatches                       = false;
			      normalizeMethod                     = "none";
			      dWMax                               = 1.0; --0.5; --0.01;
			      normalizeDw                         = false;
			      keepKernelsSynchronized             = true;
			      useMask                             = false;
			   }
	       )
	       if S1_Movie then
		  pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error"].triggerLayerName = "GroundTruth" .. i_frame-1;
		  pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error"].triggerOffset = 0;
	       end
	       if checkpointID_SLP then
		  pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error"].weightInitType       = "FileWeight";
		  pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error"].initWeightsFile
		     = inputPathSLP .. "/Checkpoints/Checkpoint" .. checkpointID_SLP .. "/S1_" .. 0 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error_W.pvp";
		  pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error"].useListOfArborFiles  = false;
		  pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error"].combineWeightFiles   = false;
	       end -- checkpointID_SLP

	       if not plasticityFlag then
		  pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error"].plasticityFlag = false;
		  pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error"].triggerLayerName = NULL;
		  pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error"].triggerOffset = nil;
		  pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error"].triggerBehavior = nil;
		  pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error"].dWMax = nil;
	       end -- not plasticityFlag

	    else -- i_delay > 1

	       pv.addGroup(pvParams,
			   "S1_" .. i_delay-1 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error",

			   {
			      groupType                           = "PlasticCloneConn";
			      preLayerName                        = "S1_" .. i_delay-1;
			      postLayerName                       = "GroundTruth" .. i_frame-1 .. "ReconS1Error";
			      channelCode                         = -1;
			      delay                               = {0.000000};
			      selfFlag                            = false;
			      preActivityIsNotRate                = false;
			      updateGSynFromPostPerspective       = false;
			      pvpatchAccumulateType               = "convolve";
			      originalConnName                    = "S1_" .. 0 .. "To" .. "GroundTruth" .. delta_frame .. "ReconS1Error";
			   }
	       )

	    end -- i_delay == 1
	 end -- delta_frame >= 0
      end -- i_delay 

      if i_frame == 1 then
	 pv.addGroup(pvParams, "BiasS1" .. "To" .. "GroundTruth" .. 0 .. "ReconS1Error",
		     pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. 0 .. "ReconS1Error"],
		     {
			preLayerName                        = "BiasS1";
			dWMax                               = 0.01;
		     }
	 )

      if not plasticityFlag then
	 pvParams["BiasS1ToGroundTruth" .. 0 .. "ReconS1Error"].triggerLayerName    = NULL;
	 pvParams["BiasS1ToGroundTruth" .. 0 .. "ReconS1Error"].triggerOffset       = nil;
	 pvParams["BiasS1ToGroundTruth" .. 0 .. "ReconS1Error"].triggerBehavior     = nil;
	 pvParams["BiasS1ToGroundTruth" .. 0 .. "ReconS1Error"].dWMax               = nil;
      end
      if checkpointID_SLP then
	 pvParams["BiasS1ToGroundTruth" .. 0 .. "ReconS1Error"].weightInitType      = "FileWeight";
	 pvParams["BiasS1ToGroundTruth" .. 0 .. "ReconS1Error"].initWeightsFile
	    = inputPathSLP .. "/Checkpoints/Checkpoint" .. checkpointID_SLP .. "/BiasS1ToGroundTruth" .. 0 .. "ReconS1Error_W.pvp";
	 pvParams["BiasS1ToGroundTruth" .. 0 .. "ReconS1Error"].useListOfArborFiles = false;
	 pvParams["BiasS1ToGroundTruth" .. 0 .. "ReconS1Error"].combineWeightFiles  = false;    
      end -- checkpointID_SLP
      
      else
	 pv.addGroup(pvParams, "BiasS1" .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1Error",
		     pvParams["S1_" .. 1 .. "To" .. "GroundTruth" .. 1 .. "ReconS1Error"],
		     {
			preLayerName                        = "BiasS1";
			postLayerName                       = "GroundTruth" .. i_frame-1 .. "ReconS1Error";
			originalConnName                    = "BiasS1ToGroundTruth" .. 0 .. "ReconS1Error";
		     }
	 )
      end
   end -- i_frame
   
   
   for i_frame = 1, numFrames do 
      for i_delay = 1, numFrames - temporalKernelSize + 1 do
	 local delta_frame = i_frame - i_delay
	 if (delta_frame >= 0 and delta_frame < temporalKernelSize) then        
	    
	    pv.addGroup(pvParams,
			"S1_" .. i_delay-1 .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1_" .. i_delay-1,
			{
			   groupType                           = "CloneConn";
			   preLayerName                        = "S1_" .. i_delay-1;
			   postLayerName                       = "GroundTruth" .. i_frame-1 .. "ReconS1_" .. i_delay-1;
			   originalConnName                    = "S1_" .. i_delay-1 .. "ToGroundTruth" .. i_frame-1 .. "ReconS1Error";
			   channelCode                         = 0;
			   delay                               = {0.000000};
			   convertRateToSpikeCount             = false;
			   receiveGpu                          = true; --false;
			   updateGSynFromPostPerspective       = true; --false;
			   pvpatchAccumulateType               = "convolve";
			   writeStep                           = -1;
			   writeCompressedCheckpoints          = false;
			   selfFlag                            = false;
			}
	    )
	    	    
	 end -- delta_frame >= 0
      end -- i_delay

      pv.addGroup(pvParams,
		  "BiasS1" .. "To" .. "GroundTruth" .. i_frame-1 .. "ReconS1",
		  pvParams["S1_" .. 0 .. "To" .. "GroundTruth" .. 0 .. "ReconS1_" .. 0],
		  {
		     preLayerName                        = "BiasS1";
		     postLayerName                       = "GroundTruth" .. i_frame-1 .. "ReconS1";
		     originalConnName                    = "BiasS1ToGroundTruth" .. 0 .. "ReconS1Error";
		  }
      )
      
   end -- i_frame
      
end -- GroundTruthPath

-- Energy probe

if not S1_Movie then

   pv.addGroup(pvParams, "S1EnergyProbe", 
	       {
		  groupType                           = "ColumnEnergyProbe";
		  probeOutputFile                     = "S1EnergyProbe.txt";
	       }
   )
   
   for i_frame = 1, numFrames do  

      pv.addGroup(pvParams, "FrameLeft" .. i_frame-1 .. "ReconS1ErrorEnergyProbe",
		  {
		     groupType                           = "L2NormProbe";
		     targetLayer                         = "FrameLeft" .. i_frame-1 .. "ReconS1Error";
		     message                             = NULL;
		     textOutputFlag                      = true;
		     probeOutputFile                     = "FrameLeft" .. i_frame-1 .. "ReconS1ErrorEnergyProbe.txt";
		     triggerLayerName                    = NULL; --"FrameLeft0";
		     --triggerOffset                       = 1;
		     energyProbe                         = "S1EnergyProbe";
		     coefficient                         = 0.5;
		     maskLayerName                       = NULL;
		     exponent                            = 2;
		  }
      )
      pv.addGroup(pvParams, "FrameRight" .. i_frame-1 .. "ReconS1ErrorEnergyProbe",
		  pvParams["FrameLeft" .. i_frame-1 .. "ReconS1ErrorEnergyProbe"],
		  {
		     targetLayer                         = "FrameRight" .. i_frame-1 .. "ReconS1Error";
		     probeOutputFile                     = "FrameRight" .. i_frame-1 .. "ReconS1ErrorEnergyProbe.txt";
		  }
      )

   end -- i_frame

   for i_delay = 1, numFrames - temporalKernelSize + 1 do
      pv.addGroup(pvParams, "S1_" .. i_delay-1 .. "SparsityProbe",
		  {
		     groupType                           = "FirmThresholdCostFnLCAProbe";
		     targetLayer                         = "S1_" .. i_delay-1;
		     message                             = NULL;
		     textOutputFlag                      = true;
		     probeOutputFile                     = "S1_" .. i_delay-1 .. "SparsityProbe.txt";
		     triggerLayerName                    = NULL;
		     energyProbe                         = "S1EnergyProbe";
		     maskLayerName                       = NULL;
		  }
      )
   end -- i_delay

   
   
end -- not S1_Movie


-- Print out PetaVision approved parameter file to the console
pv.printConsole(pvParams)
