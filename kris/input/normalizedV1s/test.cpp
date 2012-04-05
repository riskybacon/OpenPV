HyPerCol "column" = {
    //nx = 256;
    //ny = 256;
    nx = 1920;
    ny = 1080;
    //nx = 1080;
    //ny = 1920;
    dt = 1;
    randomSeed = 1;
    numSteps = 50;
    //outputPath = "/home/kpeterson/competitionwork/tower/output/";
    outputPath = "/home/kpeterson/competitionwork/heli/output/";
    progressStep=2000;
    filenamesContainLayerNames=0;
    checkpointRead=0;
    checkpointWrite=0;
    //outputPath = "/Users/kpeterson/Documents/testrun1st200/";
    //outputPath = "/Users/kpeterson/Documents/Tower/";
};
// Movie "Patterns" = {
//     //imageListPath = "/Users/kpeterson/matlabplay/filenames.txt";
//     //imageListPath = "/Users/kpeterson/Documents/Tower/filenames.txt";
//     //imageListPath = "/Users/kpeterson/Documents/heli/050/filenames.txt";
//     imageListPath = "/mnt/data1/repo/neovision-data-challenge-heli/Heli-PNG/026/filenames000.txt";
//     //imageListPath = "/mnt/data1/repo/neovision-data-challenge-tower/Tower-PNG/050/filenames000.txt";
//     restart = 0;
//     nxScale = 1;
//     nyScale = 1;
//     nf = 1;
//     marginWidth = 0;
//     writeStep = -1;
//     mirrorBCflag = 1;
//     spikingFlag = 0;
//     writeNonspikingActivity = -1;
//     
//     writeImages = 0;
// 
//     displayPeriod = 1;
//     jitterFlag = 0;
//     randomMovie = 0;
//     offsetX = 0;
//     offsetY = 0;
// 
// };

Patterns "Patterns" = {
   //marginWidth = 2;
   width = 6;
   height = 6;
   patternType = "SINEWAVE";
   orientation = "VERTICAL";
   movementType = "MOVEFORWARD";
   movementSpeed = -1;
   //movementSpeed = -6.7863;
   marginWidth = 0;

	//rotation = -2.356194490192345;
	//rotation = 0.785398163397448;
	rotation = 0;

	nxScale = 1;
	nyScale = 1;
	nf = 1;
	writeStep = 1.0; 
	mirrorBCflag = 0;
	spikingFlag =  false; // (no quotes) is translated to 0
	writeNonspikingActivity = true; // true (no quotes) is translated to 1

   
   pMove = 1; //.06=8ms//0.03125;  // probability of moving (tau ~= 1/pMove = 50 ms)
   pSwitch = 0.0;    // switch between horizontal and vertical
   writeImages = 1;  // write out image file when changes   
   //writeStep = 1;
   
   restart=0;
   offsetX=0;
   offsetY=0;
   patternsOutputPath="output/normalizedV1s/";
   writePosition=0;
};

Patterns "Patterns2" = {
   //marginWidth = 2;
   width = 6;
   height = 6;
   patternType = "SINEWAVE";
   orientation = "HORIZONTAL";
   movementType = "MOVEFORWARD";
   movementSpeed = -1;
   //movementSpeed = -6.7863;
   marginWidth = 0;

	//rotation = -2.356194490192345;
	//rotation = 0.785398163397448;
	rotation = 0;

	nxScale = 1;
	nyScale = 1;
	nf = 1;
	writeStep = 1.0; 
	mirrorBCflag = 0;
	spikingFlag =  false; // (no quotes) is translated to 0
	writeNonspikingActivity = true; // true (no quotes) is translated to 1

   
   pMove = 1; //.06=8ms//0.03125;  // probability of moving (tau ~= 1/pMove = 50 ms)
   pSwitch = 0.0;    // switch between horizontal and vertical
   writeImages = 1;  // write out image file when changes   
   //writeStep = 1;
   
   restart=0;
   offsetX=0;
   offsetY=0;
   patternsOutputPath="output/normalizedV1s/";
   writePosition=0;
};


Retina "Retina" = {
    restart = 0;
    nxScale = 1;
    nyScale = 1;
    nf = 1;
    marginWidth = 37.5;
    writeStep = 1.0;
    mirrorBCflag = 1;
    spikingFlag =  false; // (no quotes) is translated to 0
    writeNonspikingActivity = true; // true (no quotes) is translated to 1

    poissonEdgeProb = 1;
    poissonBlankProb = 0;
    burstFreq = 1;
    burstDuration = 1000;

    beginStim = 1;
    endStim = 1000;
    
};

//Image connections
IdentConn "Patterns to Retina" = {
    preLayerName = "Patterns";
    postLayerName = "Retina";
    channelCode = 0;
    weightInitType = "IdentWeight";
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;
};
IdentConn "Patterns2 to Retina" = {
    preLayerName = "Patterns2";
    postLayerName = "Retina";
    channelCode = 0;
    weightInitType = "IdentWeight";
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;
};

ANNLayer "DownScaled" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 1;
    //no = 8;
    marginWidth = 37;
    writeStep = 1.0;
    mirrorBCflag = 1;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};


// KernelConn "Retina to DownScaled" = {
//    preLayerName = "Retina";
//    postLayerName = "DownScaled";
//    channelCode = 0;
//    weightInitType = "OneToOneWeights";
//    nxp = 1; 
//    nyp = 1; 
//    nfp = 1;
//    
//    initFromLastFlag = 0;  // 1;  // restart
//    writeStep = -1;
// 
//    numAxonalArbors = 1;
//    
//    shrinkPatches=0;
// 
//    weightInit = 0.0025;
//        
//    normalize = 0.0;
//    symmetrizeWeights = 0;
// 
//    writeCompressedWeights = 0.0;
// 
//    stochasticReleaseFlag=0;
//    
//    
//    delay = 0;
// 
// };
KernelConn "Retina to DownScaled" = {
   preLayerName = "Retina";
   postLayerName = "DownScaled";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 7; 
   nyp = 7; 
   nfp = 1;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

   rMax  = infinity;
   rMin  = 0;
   
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   strength = 1;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;

};



ANNSquaredLayer "V1SimpleA1" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "DownScaled to V1SimpleA1 A" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleA1";
   channelCode = 0;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -1;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -3.53553390593274;
   flankShift = -3.53553390593274;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.03543402706968;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleA1 B" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleA1";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -1;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -3.53553390593274;
   flankShift = -2.53553390593274;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.01771701353484;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleA1 C" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleA1";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -1;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -3.53553390593274;
   flankShift = -4.53553390593274;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.01771701353484;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
ANNSquaredLayer "V1SimpleB1" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "DownScaled to V1SimpleB1 A" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB1";
   channelCode = 0;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -1;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -3.53553390593274;
   flankShift = -4.53553390593274;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.024895578861466;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleB1 B" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB1";
   channelCode = 0;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -1;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -3.53553390593274;
   flankShift = -1.53553390593274;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.012447789430733;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleB1 C" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB1";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -1;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -3.53553390593274;
   flankShift = -2.53553390593274;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.024895578861466;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleB1 D" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB1";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -1;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -3.53553390593274;
   flankShift = -5.53553390593274;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.012447789430733;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
ANNLayer "V1Complex1" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
IdentConn "V1SimpleA1 to V1Complex1" = {
    preLayerName = "V1SimpleA1";
    postLayerName = "V1Complex1";
    channelCode = 0;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
IdentConn "V1SimpleB1 to V1Complex1" = {
    preLayerName = "V1SimpleB1";
    postLayerName = "V1Complex1";
    channelCode = 0;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
ANNLayer "V1ComplexSum1" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "V1Complex1 to V1ComplexSum1" = {
   preLayerName = "V1Complex1";
   postLayerName = "V1ComplexSum1";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
ANNDivInhLayer "V1ComplexBar1" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
IdentConn "V1Complex1 to V1ComplexBar1" = {
    preLayerName = "V1Complex1";
    postLayerName = "V1ComplexBar1";
    channelCode = 0;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
IdentConn "V1ComplexSum1 to V1ComplexBar1" = {
    preLayerName = "V1ComplexSum1";
    postLayerName = "V1ComplexBar1";
    channelCode = 2;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
ANNSquaredLayer "V1SimpleA2" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "DownScaled to V1SimpleA2 A" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleA2";
   channelCode = 0;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.707106781186547;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.08248290463863;
   flankShift = -2.88675134594813;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 1;
       
   strength = 0.03543402706968;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleA2 B" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleA2";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.707106781186547;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.08248290463863;
   flankShift = -1.88675134594813;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 1;
       
   strength = 0.01771701353484;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleA2 C" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleA2";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.707106781186547;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.08248290463863;
   flankShift = -3.88675134594813;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 1;
       
   strength = 0.01771701353484;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
ANNSquaredLayer "V1SimpleB2" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "DownScaled to V1SimpleB2 A" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB2";
   channelCode = 0;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.707106781186547;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.08248290463863;
   flankShift = -3.88675134594813;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 1;
       
   strength = 0.024895578861466;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleB2 B" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB2";
   channelCode = 0;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.707106781186547;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.08248290463863;
   flankShift = -0.886751345948127;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 1;
       
   strength = 0.012447789430733;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleB2 C" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB2";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.707106781186547;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.08248290463863;
   flankShift = -1.88675134594813;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 1;
       
   strength = 0.024895578861466;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleB2 D" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB2";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.707106781186547;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.08248290463863;
   flankShift = -4.88675134594813;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 1;
       
   strength = 0.012447789430733;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
ANNLayer "V1Complex2" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
IdentConn "V1SimpleA2 to V1Complex2" = {
    preLayerName = "V1SimpleA2";
    postLayerName = "V1Complex2";
    channelCode = 0;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
IdentConn "V1SimpleB2 to V1Complex2" = {
    preLayerName = "V1SimpleB2";
    postLayerName = "V1Complex2";
    channelCode = 0;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
ANNLayer "V1ComplexSum2" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "V1Complex2 to V1ComplexSum2" = {
   preLayerName = "V1Complex2";
   postLayerName = "V1ComplexSum2";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
ANNDivInhLayer "V1ComplexBar2" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
IdentConn "V1Complex2 to V1ComplexBar2" = {
    preLayerName = "V1Complex2";
    postLayerName = "V1ComplexBar2";
    channelCode = 0;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
IdentConn "V1ComplexSum2 to V1ComplexBar2" = {
    preLayerName = "V1ComplexSum2";
    postLayerName = "V1ComplexBar2";
    channelCode = 2;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
ANNSquaredLayer "V1SimpleA3" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "DownScaled to V1SimpleA3 A" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleA3";
   channelCode = 0;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 1;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = 0;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -5;
   flankShift = 0;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.03543402706968;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleA3 B" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleA3";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 1;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = 0;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -5;
   flankShift = 2;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.01771701353484;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleA3 C" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleA3";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 1;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = 0;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -5;
   flankShift = -2;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.01771701353484;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
ANNSquaredLayer "V1SimpleB3" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "DownScaled to V1SimpleB3 A" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB3";
   channelCode = 0;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 1;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = 0;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -5;
   flankShift = -2;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.024895578861466;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleB3 B" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB3";
   channelCode = 0;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 1;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = 0;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -5;
   flankShift = 4;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.012447789430733;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleB3 C" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB3";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 1;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = 0;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -5;
   flankShift = 2;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.024895578861466;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleB3 D" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB3";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 1;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = 0;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -5;
   flankShift = -4;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.012447789430733;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
ANNLayer "V1Complex3" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
IdentConn "V1SimpleA3 to V1Complex3" = {
    preLayerName = "V1SimpleA3";
    postLayerName = "V1Complex3";
    channelCode = 0;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
IdentConn "V1SimpleB3 to V1Complex3" = {
    preLayerName = "V1SimpleB3";
    postLayerName = "V1Complex3";
    channelCode = 0;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
ANNLayer "V1ComplexSum3" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "V1Complex3 to V1ComplexSum3" = {
   preLayerName = "V1Complex3";
   postLayerName = "V1ComplexSum3";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
ANNDivInhLayer "V1ComplexBar3" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
IdentConn "V1Complex3 to V1ComplexBar3" = {
    preLayerName = "V1Complex3";
    postLayerName = "V1ComplexBar3";
    channelCode = 0;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
IdentConn "V1ComplexSum3 to V1ComplexBar3" = {
    preLayerName = "V1ComplexSum3";
    postLayerName = "V1ComplexBar3";
    channelCode = 2;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
ANNSquaredLayer "V1SimpleA4" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "DownScaled to V1SimpleA4 A" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleA4";
   channelCode = 0;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.5;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.47213595499958;
   flankShift = -2.23606797749979;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.03543402706968;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleA4 B" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleA4";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.5;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.47213595499958;
   flankShift = -1.23606797749979;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.01771701353484;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleA4 C" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleA4";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.5;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.47213595499958;
   flankShift = -3.23606797749979;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.01771701353484;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
ANNSquaredLayer "V1SimpleB4" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "DownScaled to V1SimpleB4 A" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB4";
   channelCode = 0;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.5;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.47213595499958;
   flankShift = -3.23606797749979;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.024895578861466;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleB4 B" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB4";
   channelCode = 0;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.5;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.47213595499958;
   flankShift = -0.23606797749979;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.012447789430733;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleB4 C" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB4";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.5;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.47213595499958;
   flankShift = -1.23606797749979;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.024895578861466;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
KernelConn "DownScaled to V1SimpleB4 D" = {
   preLayerName = "DownScaled";
   postLayerName = "V1SimpleB4";
   channelCode = 1;
   weightInitType = "Windowed3DGaussWeights";
   nxp = 15; 
   nyp = 15; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   dT = 1;

   rMax  = infinity;
   deltaThetaMax = 6.2832;
   thetaMax = 2;
   bowtieFlag = 0;
   numFlanks = 1;
   
   flowSpeed = -0.5;
   yaspect = 10;
   taspect = 1;
   sigma = 20;
   shiftT = -4.47213595499958;
   flankShift = -4.23606797749979;
   windowShiftT=-5;
   windowShift=0;
   numAxonalArbors = 11;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

   rotate = 0;
       
   strength = 0.012447789430733;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 1.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   
   
   delay = 0;
   normalize_arbors_individually=0;

};
ANNLayer "V1Complex4" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
IdentConn "V1SimpleA4 to V1Complex4" = {
    preLayerName = "V1SimpleA4";
    postLayerName = "V1Complex4";
    channelCode = 0;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
IdentConn "V1SimpleB4 to V1Complex4" = {
    preLayerName = "V1SimpleB4";
    postLayerName = "V1Complex4";
    channelCode = 0;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
ANNLayer "V1ComplexSum4" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "V1Complex4 to V1ComplexSum4" = {
   preLayerName = "V1Complex4";
   postLayerName = "V1ComplexSum4";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
ANNDivInhLayer "V1ComplexBar4" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
IdentConn "V1Complex4 to V1ComplexBar4" = {
    preLayerName = "V1Complex4";
    postLayerName = "V1ComplexBar4";
    channelCode = 0;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
IdentConn "V1ComplexSum4 to V1ComplexBar4" = {
    preLayerName = "V1ComplexSum4";
    postLayerName = "V1ComplexBar4";
    channelCode = 2;
    writeStep = -1.0;
    delay = 0;
    initFromLastFlag = 0;

};
KernelConn "V1Complex1 to V1ComplexSum2" = {
   preLayerName = "V1Complex1";
   postLayerName = "V1ComplexSum2";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
KernelConn "V1Complex1 to V1ComplexSum3" = {
   preLayerName = "V1Complex1";
   postLayerName = "V1ComplexSum3";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
KernelConn "V1Complex1 to V1ComplexSum4" = {
   preLayerName = "V1Complex1";
   postLayerName = "V1ComplexSum4";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
KernelConn "V1Complex2 to V1ComplexSum1" = {
   preLayerName = "V1Complex2";
   postLayerName = "V1ComplexSum1";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
KernelConn "V1Complex2 to V1ComplexSum3" = {
   preLayerName = "V1Complex2";
   postLayerName = "V1ComplexSum3";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
KernelConn "V1Complex2 to V1ComplexSum4" = {
   preLayerName = "V1Complex2";
   postLayerName = "V1ComplexSum4";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
KernelConn "V1Complex3 to V1ComplexSum1" = {
   preLayerName = "V1Complex3";
   postLayerName = "V1ComplexSum1";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
KernelConn "V1Complex3 to V1ComplexSum2" = {
   preLayerName = "V1Complex3";
   postLayerName = "V1ComplexSum2";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
KernelConn "V1Complex3 to V1ComplexSum4" = {
   preLayerName = "V1Complex3";
   postLayerName = "V1ComplexSum4";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
KernelConn "V1Complex4 to V1ComplexSum1" = {
   preLayerName = "V1Complex4";
   postLayerName = "V1ComplexSum1";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
KernelConn "V1Complex4 to V1ComplexSum2" = {
   preLayerName = "V1Complex4";
   postLayerName = "V1ComplexSum2";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
KernelConn "V1Complex4 to V1ComplexSum3" = {
   preLayerName = "V1Complex4";
   postLayerName = "V1ComplexSum3";
   channelCode = 0;
   weightInitType = "Gauss2DWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;
   rMax  = infinity;
   rMin  = 0;
   aspect = 1;
   sigma = 5;
   numAxonalArbors = 1;
   shrinkPatches=0;
   stochasticReleaseFlag=0;
   strength = 4;  
   normalize = 1.0;
   normalize_zero_offset = 0.0;
   normalize_max = 0.0;
   symmetrizeWeights = 0;
   writeCompressedWeights = 0.0;
   normalize_cutoff = 0;
   plasticityFlag = 0;
   delay = 0;

};
ANNLayer "MTLayer1" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "V1ComplexBar1 to MTLayer1 Exh" = {
   preLayerName = "V1ComplexBar1";
   postLayerName = "MTLayer1";
   channelCode = 0;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -1;
   rotate = 0;
   thetaMax = 2;
   inputV1Speed = -1;
   inputV1Rotate=0;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar1 to MTLayer1 Inh" = {
   preLayerName = "V1ComplexBar1";
   postLayerName = "MTLayer1";
   channelCode = 1;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -1;
   rotate = 0;
   thetaMax = 2;
   inputV1Speed = -1;
   inputV1Rotate=0;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar2 to MTLayer1 Exh" = {
   preLayerName = "V1ComplexBar2";
   postLayerName = "MTLayer1";
   channelCode = 0;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -1;
   rotate = 0;
   thetaMax = 2;
   inputV1Speed = -0.707106781186547;
   inputV1Rotate=1;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar2 to MTLayer1 Inh" = {
   preLayerName = "V1ComplexBar2";
   postLayerName = "MTLayer1";
   channelCode = 1;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -1;
   rotate = 0;
   thetaMax = 2;
   inputV1Speed = -0.707106781186547;
   inputV1Rotate=1;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar3 to MTLayer1 Exh" = {
   preLayerName = "V1ComplexBar3";
   postLayerName = "MTLayer1";
   channelCode = 0;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -1;
   rotate = 0;
   thetaMax = 2;
   inputV1Speed = 0;
   inputV1Rotate=0;
   inputV1ThetaMax=1;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar3 to MTLayer1 Inh" = {
   preLayerName = "V1ComplexBar3";
   postLayerName = "MTLayer1";
   channelCode = 1;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -1;
   rotate = 0;
   thetaMax = 2;
   inputV1Speed = 0;
   inputV1Rotate=0;
   inputV1ThetaMax=1;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
ANNLayer "MTLayer2" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "V1ComplexBar1 to MTLayer2 Exh" = {
   preLayerName = "V1ComplexBar1";
   postLayerName = "MTLayer2";
   channelCode = 0;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -0.707106781186547;
   rotate = 1;
   thetaMax = 2;
   inputV1Speed = -1;
   inputV1Rotate=0;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar1 to MTLayer2 Inh" = {
   preLayerName = "V1ComplexBar1";
   postLayerName = "MTLayer2";
   channelCode = 1;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -0.707106781186547;
   rotate = 1;
   thetaMax = 2;
   inputV1Speed = -1;
   inputV1Rotate=0;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar2 to MTLayer2 Exh" = {
   preLayerName = "V1ComplexBar2";
   postLayerName = "MTLayer2";
   channelCode = 0;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -0.707106781186547;
   rotate = 1;
   thetaMax = 2;
   inputV1Speed = -0.707106781186547;
   inputV1Rotate=1;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar2 to MTLayer2 Inh" = {
   preLayerName = "V1ComplexBar2";
   postLayerName = "MTLayer2";
   channelCode = 1;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -0.707106781186547;
   rotate = 1;
   thetaMax = 2;
   inputV1Speed = -0.707106781186547;
   inputV1Rotate=1;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar3 to MTLayer2 Exh" = {
   preLayerName = "V1ComplexBar3";
   postLayerName = "MTLayer2";
   channelCode = 0;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -0.707106781186547;
   rotate = 1;
   thetaMax = 2;
   inputV1Speed = 0;
   inputV1Rotate=0;
   inputV1ThetaMax=1;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar3 to MTLayer2 Inh" = {
   preLayerName = "V1ComplexBar3";
   postLayerName = "MTLayer2";
   channelCode = 1;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -0.707106781186547;
   rotate = 1;
   thetaMax = 2;
   inputV1Speed = 0;
   inputV1Rotate=0;
   inputV1ThetaMax=1;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar4 to MTLayer2 Exh" = {
   preLayerName = "V1ComplexBar4";
   postLayerName = "MTLayer2";
   channelCode = 0;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -0.707106781186547;
   rotate = 1;
   thetaMax = 2;
   inputV1Speed = -0.5;
   inputV1Rotate=0;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar4 to MTLayer2 Inh" = {
   preLayerName = "V1ComplexBar4";
   postLayerName = "MTLayer2";
   channelCode = 1;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = -0.707106781186547;
   rotate = 1;
   thetaMax = 2;
   inputV1Speed = -0.5;
   inputV1Rotate=0;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
ANNLayer "MTLayer3" = {
    restart = 0;
    nxScale = 0.25;
    nyScale = 0.25;
    nf = 4;
    //no = 8;
    marginWidth = 3;
    writeStep = 1.0;
    mirrorBCflag = 0;
    spikingFlag = 0;
    writeNonspikingActivity = 1;

    Vrest = 0.0;

    VThresh = -infinity;  // infinity (no quotes) is translated to FLT_MAX
    VMax = infinity;
    VMin = -infinity;
};
KernelConn "V1ComplexBar1 to MTLayer3 Exh" = {
   preLayerName = "V1ComplexBar1";
   postLayerName = "MTLayer3";
   channelCode = 0;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = 0;
   rotate = 0;
   thetaMax = 1;
   inputV1Speed = -1;
   inputV1Rotate=0;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar1 to MTLayer3 Inh" = {
   preLayerName = "V1ComplexBar1";
   postLayerName = "MTLayer3";
   channelCode = 1;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = 0;
   rotate = 0;
   thetaMax = 1;
   inputV1Speed = -1;
   inputV1Rotate=0;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar2 to MTLayer3 Exh" = {
   preLayerName = "V1ComplexBar2";
   postLayerName = "MTLayer3";
   channelCode = 0;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = 0;
   rotate = 0;
   thetaMax = 1;
   inputV1Speed = -0.707106781186547;
   inputV1Rotate=1;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar2 to MTLayer3 Inh" = {
   preLayerName = "V1ComplexBar2";
   postLayerName = "MTLayer3";
   channelCode = 1;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = 0;
   rotate = 0;
   thetaMax = 1;
   inputV1Speed = -0.707106781186547;
   inputV1Rotate=1;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar3 to MTLayer3 Exh" = {
   preLayerName = "V1ComplexBar3";
   postLayerName = "MTLayer3";
   channelCode = 0;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = 0;
   rotate = 0;
   thetaMax = 1;
   inputV1Speed = 0;
   inputV1Rotate=0;
   inputV1ThetaMax=1;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar3 to MTLayer3 Inh" = {
   preLayerName = "V1ComplexBar3";
   postLayerName = "MTLayer3";
   channelCode = 1;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = 0;
   rotate = 0;
   thetaMax = 1;
   inputV1Speed = 0;
   inputV1Rotate=0;
   inputV1ThetaMax=1;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar4 to MTLayer3 Exh" = {
   preLayerName = "V1ComplexBar4";
   postLayerName = "MTLayer3";
   channelCode = 0;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = 0;
   rotate = 0;
   thetaMax = 1;
   inputV1Speed = -0.5;
   inputV1Rotate=0;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
KernelConn "V1ComplexBar4 to MTLayer3 Inh" = {
   preLayerName = "V1ComplexBar4";
   postLayerName = "MTLayer3";
   channelCode = 1;
   weightInitType = "MTWeight";
   nxp = 1; 
   nyp = 1; 
   nfp = 4;
   
   initFromLastFlag = 0;  // 1;  // restart
   writeStep = -1;

      
   deltaThetaMax = 6.2832;
   tunedSpeed = 0;
   rotate = 0;
   thetaMax = 1;
   inputV1Speed = -0.5;
   inputV1Rotate=0;
   inputV1ThetaMax=2;
   
   numAxonalArbors = 1;
   
   shrinkPatches=0;
   stochasticReleaseFlag=0;

       
   normalize = 0.0;
   symmetrizeWeights = 0;

   writeCompressedWeights = 0.0;
   plasticityFlag = 0;
   
   
   delay = 0;

};
