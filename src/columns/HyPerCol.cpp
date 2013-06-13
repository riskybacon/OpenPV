/*
 * HyPerCol.cpp
 *
 *  Created on: Jul 30, 2008
 *      Author: Craig Rasmussen
 */

#define TIMER_ON
#define TIMESTEP_OUTPUT

#include "HyPerCol.hpp"
#include "InterColComm.hpp"
#include "../io/clock.h"
#include "../io/imageio.hpp"
#include "../io/io.h"
#include "../utils/pv_random.h"

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <float.h>

#define PV_MAX_NUMSTEPS (pow(2,DBL_MANT_DIG))
// Commented out Nov 4, 2012.  With times declared as double instead of float, the x+1==x barrier shouldn't happen until x is somewhere in the quadrillions.
// #define HYPERCOL_DIRINDEX_MAX 99999999

namespace PV {

HyPerCol::HyPerCol(const char * name, int argc, char * argv[], PVParams * params)
         : warmStart(false), isInitialized(false)
{
   initialize_base();
   initialize(name, argc, argv, params);
}

HyPerCol::~HyPerCol()
{
   int n;

#ifdef PV_USE_OPENCL
   finalizeThreads();
#endif // PV_USE_OPENCL

   if (image_file != NULL) free(image_file);

   for (n = 0; n < numConnections; n++) {
      delete connections[n];
   }

   for (n = 0; n < numLayers; n++) {
      // TODO: check to see if finalize called
      if (layers[n] != NULL) {
         delete layers[n]; // will call *_finalize
      }
      else {
         // TODO move finalize
         // PVLayer_finalize(getCLayer(n));
      }
   }

   if (ownsParams) delete params;

   if (ownsInterColComm) {
      delete icComm;
   }
   else {
      icComm->clearPublishers();
   }

   printf("%32s: total time in %6s %10s: ", name, "column", "run    ");
   runTimer->elapsed_time();
   fflush(stdout);
   delete runTimer;

   free(connections);
   free(layers);
   free(name);
   free(outputPath);
   free(outputNamesOfLayersAndConns);
   if (checkpointWriteFlag) {
      free(checkpointWriteDir); checkpointWriteDir = NULL;
   }
}

int HyPerCol::initFinish(void)
{
   int status = 0;

   for (int i = 0; i < this->numLayers; i++) {
      status = layers[i]->initFinish();
      if (status != 0) {
         fprintf(stderr, "[%d]: HyPerCol::initFinish: ERROR condition, exiting...\n", this->columnId());
         exit(status);
      }
   }

#ifdef OBSOLETE
   // TODO - fix this to modern version?
   log_parameters(numSteps, image_file);
#endif

   isInitialized = true;

   return status;
}

#define DEFAULT_NUMSTEPS 1
int HyPerCol::initialize_base() {
   // Initialize all member variables to safe values.  They will be set to their actual values in initialize()
   numSteps = 0;
   currentStep = 0;
   layerArraySize = INITIAL_LAYER_ARRAY_SIZE;
   numLayers = 0;
   numPhases = 0;
   connectionArraySize = INITIAL_CONNECTION_ARRAY_SIZE;
   numConnections = 0;
   checkpointReadFlag = false;
   checkpointWriteFlag = false;
   checkpointReadDir = NULL;
   cpReadDirIndex = -1L;
   checkpointWriteDir = NULL;
   cpWriteStepInterval = -1L;
   nextCPWriteStep = 0L;
   cpWriteTimeInterval = -1.0;
   nextCPWriteTime = 0.0;
   deleteOlderCheckpoints = false;
   memset(lastCheckpointDir, 0, PV_PATH_MAX);
   suppressLastOutput = false;
   simTime = 0.0;
   stopTime = 0.0;
   deltaTime = DELTA_T;
   progressStep = 1L;
   writeProgressToErr = false;
   clDevice = NULL;
   layers = NULL;
   connections = NULL;
   name = NULL;
   outputPath = NULL;
   outputNamesOfLayersAndConns = NULL;
   image_file = NULL;
   nxGlobal = 0;
   nyGlobal = 0;
   ownsParams = true;
   ownsInterColComm = true;
   params = NULL;
   icComm = NULL;
   runDelegate = NULL;
   runTimer = NULL;
   numProbes = 0;
   probes = NULL;
   filenamesContainLayerNames = 0;
   filenamesContainConnectionNames = 0;
   random_seed = 0;
   random_seed_obj = 0;

   return PV_SUCCESS;
}

int HyPerCol::initialize(const char * name, int argc, char ** argv, PVParams * params)
{
   ownsInterColComm = (params==NULL || params->getInterColComm()==NULL);
   if (ownsInterColComm) {
      icComm = new InterColComm(&argc, &argv);
   }
   else {
      icComm = params->getInterColComm();
   }
   int rank = icComm->commRank();

#ifdef PVP_DEBUG
   bool reqrtn = false;
   for(int arg=1; arg<argc; arg++) {
      if( !strcmp(argv[arg], "--require-return")) {
         reqrtn = true;
         break;
      }
   }
   if( reqrtn ) {
      if( rank == 0 ) {
         printf("Hit enter to begin! ");
         fflush(stdout);
         int charhit = -1;
         while(charhit != '\n') {
            charhit = getc(stdin);
         }
      }
#ifdef PV_USE_MPI
      MPI_Barrier(icComm->communicator());
#endif // PV_USE_MPI
   }
#endif // PVP_DEBUG

   this->name = strdup(name);
   this->runTimer = new Timer();

   layers = (HyPerLayer **) malloc(layerArraySize * sizeof(HyPerLayer *));
   connections = (HyPerConn **) malloc(connectionArraySize * sizeof(HyPerConn *));

   int opencl_device = 0;  // default to GPU for now
   char * param_file = NULL;
   char * working_dir = NULL;
   parse_options(argc, argv, &outputPath, &param_file,
                 &numSteps, &opencl_device, &random_seed, &working_dir);

   if(working_dir) {
      int status = chdir(working_dir);
      if(status) {
         fprintf(stderr, "Unable to switch directory to \"%s\"\n", working_dir);
         fprintf(stderr, "chdir error: %s\n", strerror(errno));
         exit(status);
      }
   }

   // path to working directory is no longer used
   // path = (char *) malloc(1+PV_PATH_MAX);
   // assert(path != NULL);
   // path = getcwd(path, PV_PATH_MAX);

   ownsParams = params==NULL;
   if (ownsParams) {
      size_t groupArraySize = 2*(layerArraySize + connectionArraySize);
      params = new PVParams(param_file, groupArraySize, icComm);  // PVParams::addGroup can resize if initialGroups is exceeded
   }
   this->params = params;
   free(param_file);
   param_file = NULL;

#ifdef PV_USE_MPI // Fail if there was a parsing error, but make sure nonroot processes don't kill the root process before the root process reaches the syntax error
   int parsedStatus;
   int rootproc = 0;
   if( rank == rootproc ) {
      parsedStatus = params->getParseStatus();
   }
   MPI_Bcast(&parsedStatus, 1, MPI_INT, rootproc, icCommunicator()->communicator());
#else
   int parsedStatus = params->getParseStatus();
#endif
   if( parsedStatus != 0 ) {
      exit(parsedStatus);
   }

   // set number of steps from params file if it wasn't set on the command line
   if( !numSteps ) {
      if( params->present(name, "numSteps") ) {
         numSteps = (long int) params->value(name, "numSteps");
      }
      else {
         printf("Number of steps specified neither in the command line nor the params file.\n"
                "Number of steps set to default %ld\n",numSteps);
      }
   }
   if( (double) numSteps > PV_MAX_NUMSTEPS ) {
      fprintf(stderr, "The number of time steps %ld is greater than %ld, the maximum allowed by floating point precision\n", numSteps, (long int) PV_MAX_NUMSTEPS);
      exit(EXIT_FAILURE);
   }

   // set how often advanceTime() prints a message indicating progress
   progressStep       = (long int) params->value(name, "progressStep",progressStep);
   writeProgressToErr = params->value(name, "writeProgressToErr",writeProgressToErr)!=0;

   // set output path from params file if it wasn't set on the command line
   if (outputPath == NULL ) {
      if( params->stringPresent(name, "outputPath") ) {
         outputPath = strdup(params->stringValue(name, "outputPath"));
         assert(outputPath != NULL);
      }
      else {
         outputPath = strdup(OUTPUT_PATH);
         assert(outputPath != NULL);
         printf("Output path specified neither in command line nor in params file.\n"
                "Output path set to default \"%s\"\n",OUTPUT_PATH);
      }
   }
   ensureDirExists(outputPath);

   if (params->stringPresent(name, "printParamsFilename")) {
      const char * printParamsFilename = params->stringValue(name, "printParamsFilename", false);
      outputParams(printParamsFilename);
   }
   else {
      outputParams("params.pv");
   }
#ifdef UNDERCONSTRUCTION // We plan to create an XML file containing all params whether specified in params or by default value
              // The problem with doing that from within HyPerCol::initialize is that the layers and columns haven't been added yet.
   if (params->stringPresent(name, "paramsXMLFilename")) {
      const char * paramsXMLFilename = params->stringValue(name, "paramsXMLFilename", true);
      outputParamsXML(paramsXMLFilename);
   }
   else {
      outputParamsXML("params.xml");
   }
#endif // UNDERCONSTRUCTION

   // run only on GPU for now
#ifdef PV_USE_OPENCL
   initializeThreads(opencl_device);
   clDevice->query_device_info();
#endif

   // set random seed if it wasn't set in the command line
   // bool seedfromclock = false;
   if( !random_seed ) {
      if( params->present(name, "randomSeed") ) {
         random_seed = (unsigned long) params->value(name, "randomSeed");
      }
      else {
         random_seed = getRandomSeed();
         // seedfromclock = true; // Commented out Nov. 28, 2012.  getRandomSeed prints the seed so seedfromclock isn't needed
      }
   }
   if (random_seed < 10000000) {
      fprintf(stderr, "Error: random seed %lu is too small. Use a seed of at least 10000000.\n", random_seed);
      abort();
   }
   // random_seed /= (unsigned long) (1+columnId()); // Commented out Nov. 28, 2012.  For reproducibility across MPI configurations, need all processes to use the same seed

   // Commented out Nov. 28, 2012.  getRandomSeed prints the seed so we don't have to do it here.
   // if (seedfromclock) {
   //    if (icComm->commRank()==0) {
   //       printf("Using time to get random seed.\n");
   //    }
   //    printf("Rank %d process seed set to %lu\n", icComm->commRank(), random_seed);
   // }

   // Commented out Nov. 28, 2012.  Individual neurons have a Tausworthe pseudo-rng so that their state can be
   // recovered in checkpointing, and so that MPI runs are reproducible even though MPI processes call the
   // random() system call a nondeterministic number of times.
   // pv_srandom(random_seed); // initialize random seed
   random_seed_obj = random_seed;

   nxGlobal = (int) params->value(name, "nx");
   nyGlobal = (int) params->value(name, "ny");

   deltaTime = params->value(name, "dt", deltaTime, true);

   runDelegate = NULL;

   numProbes = 0;
   probes = NULL;

   filenamesContainLayerNames = (int)params->value(name, "filenamesContainLayerNames", 0);
   if(filenamesContainLayerNames < 0 || filenamesContainLayerNames > 2) {
      fprintf(stderr,"HyPerCol %s: filenamesContainLayerNames must have the value 0, 1, or 2.\n", name);
      abort();
   }

   filenamesContainConnectionNames = (int)params->value(name, "filenamesContainConnectionNames", 0);
   if(filenamesContainConnectionNames < 0 || filenamesContainConnectionNames > 2) {
      fprintf(stderr,"HyPerCol %s: filenamesContainConnectionNames must have the value 0, 1, or 2.\n", name);
      abort();
   }

   const char * lcfilename = params->stringValue(name, "outputNamesOfLayersAndConns", false);
   if( lcfilename != NULL && lcfilename[0] != 0 && rank==0 ) {
      outputNamesOfLayersAndConns = (char *) malloc( (strlen(outputPath)+strlen(lcfilename)+2)*sizeof(char) );
      if( !outputNamesOfLayersAndConns ) {
         fprintf(stderr, "HyPerCol \"%s\": Unable to allocate memory for outputNamesOfLayersAndConns.  Exiting.\n", name);
         exit(EXIT_FAILURE);
      }
      sprintf(outputNamesOfLayersAndConns, "%s/%s", outputPath, lcfilename);
   }
   else {
      outputNamesOfLayersAndConns = NULL;
   }

   checkpointReadFlag = params->value(name, "checkpointRead", false) != 0;
   if(checkpointReadFlag) {
      const char * cpreaddir = params->stringValue(name, "checkpointReadDir", true);
      if( cpreaddir != NULL ) {
         checkpointReadDir = strdup(cpreaddir);
      }
      else {
         fprintf(stderr, "Rank %d: Column \"%s\": if checkpointRead is set, the string checkpointReadDir must be defined.  Exiting.\n", rank, name);
         exit(EXIT_FAILURE);
      }
      struct stat checkpointReadDirStat;
      int dirExistStatus = checkDirExists(checkpointReadDir, &checkpointReadDirStat);
      if( dirExistStatus != 0 ) {
         fprintf(stderr, "Rank %d: Column \"%s\": unable to read checkpointReadDir \"%s\": %s\n", rank, name, checkpointReadDir, strerror(dirExistStatus));
         exit(EXIT_FAILURE);
      }
      cpReadDirIndex = (int) params->value(name, "checkpointReadDirIndex", -1, true);

      if (cpReadDirIndex < 0) {
         fflush(stdout);
         fprintf(stderr, "Rank %d: Column \"%s\": checkpointReadDirIndex must be nonnegative", rank, name);
      }
// Commented out Nov 4, 2012
//      if (cpReadDirIndex < 0 || cpReadDirIndex > HYPERCOL_DIRINDEX_MAX ) {
//            fflush(stdout);
//            fprintf(stderr, "Rank %d: Column \"%s\": checkpointReadDirIndex must be between 0 and %d, inclusive.  Exiting.\n", rank, name, HYPERCOL_DIRINDEX_MAX);
//         exit(EXIT_FAILURE);
//      }
   }

   checkpointWriteFlag = params->value(name, "checkpointWrite", false) != 0;
   if(checkpointWriteFlag) {
      const char * cpwritedir = params->stringValue(name, "checkpointWriteDir", true);
      if( cpwritedir != NULL ) {
         checkpointWriteDir = strdup(cpwritedir);
      }
      else {
         if( rank == 0 ) {
            fprintf(stderr, "Column \"%s\": if checkpointWrite is set, the string checkpointWriteDir must be defined.  Exiting.\n", name);
         }
         exit(EXIT_FAILURE);
      }
      ensureDirExists(checkpointWriteDir);

      char cpDir[PV_PATH_MAX];
      int chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Checkpoint%ld", checkpointWriteDir, numSteps);
      if(chars_printed >= PV_PATH_MAX) {
         if (icComm->commRank()==0) {
            fprintf(stderr,"HyPerCol::run error.  Checkpoint directory \"%s/Checkpoint%ld\" will be needed to hold the checkpoint at end of run, but this path is too long.\n", checkpointWriteDir, numSteps);
            abort();
         }
      }

      bool usingWriteStep = params->present(name, "checkpointWriteStepInterval") && params->value(name, "checkpointWriteStepInterval")>0;
      bool usingWriteTime = params->present(name, "checkpointWriteTimeInterval") && params->value(name, "checkpointWriteTimeInterval")>0;
      if( !usingWriteStep && !usingWriteTime ) {
         if( rank == 0 ) {
            fflush(stdout);
            fprintf(stderr,"If checkpointWrite is set, one of checkpointWriteStepInterval or checkpointWriteTimeInterval must be positive.\n");
         }
         exit(EXIT_FAILURE);
      }
      if( usingWriteStep && usingWriteTime ) {
         if( rank == 0 ) {
            fflush(stdout);
            fprintf(stderr,"If checkpointWrite is set, only one of checkpointWriteStepInterval or checkpointWriteTimeInterval can be positive.\n");
         }
         exit(EXIT_FAILURE);
      }
      if( usingWriteStep ) {
         cpWriteStepInterval = (long int) params->value(name, "checkpointWriteStepInterval");
         cpWriteTimeInterval = -1;
      }
      if( usingWriteTime ) {
         cpWriteTimeInterval = params->value(name, "checkpointWriteTimeInterval");
         cpWriteStepInterval = -1;
      }
      nextCPWriteStep = 0;
      nextCPWriteTime = 0;

      deleteOlderCheckpoints = params->value(name, "deleteOlderCheckpoints", false) != 0;
      if (deleteOlderCheckpoints) {
         memset(lastCheckpointDir, 0, PV_PATH_MAX);
      }
   }
   else {
      checkpointWriteDir = NULL;
      suppressLastOutput = params->value(name, "suppressLastOutput", false, true) != 0;
   }

   return PV_SUCCESS;
}

int HyPerCol::checkDirExists(const char * dirname, struct stat * pathstat) {
   // check if the given directory name exists for the rank zero process
   // the return value is zero if a successful stat(2) call and the error
   // if unsuccessful.  pathstat contains the result of the buffer from the stat call.
   // The rank zero process is the only one that calls stat(); it then Bcasts the
   // result to the rest of the processes.
   assert(pathstat);

   int rank = icComm->commRank();
   int status;
   int errorcode;
   if( rank == 0 ) {
      status = stat(dirname, pathstat);
      if( status ) errorcode = errno;
   }
#ifdef PV_USE_MPI
   MPI_Bcast(&status, 1, MPI_INT, 0, icCommunicator()->communicator());
   if( status ) {
      MPI_Bcast(&errorcode, 1, MPI_INT, 0, icCommunicator()->communicator());
   }
   MPI_Bcast(pathstat, sizeof(struct stat), MPI_CHAR, 0, icCommunicator()->communicator());
#endif // PV_USE_MPI
   return status ? errorcode : 0;
}

int HyPerCol::ensureDirExists(const char * dirname) {
   // see if path exists, and try to create it if it doesn't.
   // Since only rank 0 process should be reading and writing, only rank 0 does the mkdir call
   int rank = icComm->commRank();
   struct stat pathstat;
   int resultcode = checkDirExists(dirname, &pathstat);
   if( resultcode == 0 ) { // outputPath exists; now check if it's a directory.
      if( !(pathstat.st_mode & S_IFDIR ) ) {
         if( rank == 0 ) {
            fflush(stdout);
            fprintf(stderr, "Path \"%s\" exists but is not a directory\n", dirname);
         }
         exit(EXIT_FAILURE);
      }
   }
   else if( resultcode == ENOENT /* No such file or directory */ ) {
      if( rank == 0 ) {
         printf("Directory \"%s\" does not exist; attempting to create\n", dirname);

         char targetString[PV_PATH_MAX];
         int num_chars_needed = snprintf(targetString,PV_PATH_MAX,"mkdir -p %s",dirname);
         if (num_chars_needed > PV_PATH_MAX) {
            fflush(stdout);
            fprintf(stderr,"Path \"%s\" is too long.",dirname);
            exit(EXIT_FAILURE);
         }
         int mkdirstatus = system(targetString);
         if( mkdirstatus != 0 ) {
            fflush(stdout);
            fprintf(stderr, "Directory \"%s\" could not be created: %s\n", dirname, strerror(errno));
            exit(EXIT_FAILURE);
         }
      }
   }
   else {
      if( rank == 0 ) {
         fflush(stdout);
         fprintf(stderr, "Error checking status of directory \"%s\": %s\n", dirname, strerror(resultcode));
      }
      exit(EXIT_FAILURE);
   }
   return PV_SUCCESS;
}

int HyPerCol::columnId()
{
   return icComm->commRank();
}

int HyPerCol::numberOfColumns()
{
   return icComm->numCommRows() * icComm->numCommColumns();
}

int HyPerCol::commColumn(int colId)
{
   return colId % icComm->numCommColumns();
}

int HyPerCol::commRow(int colId)
{
   return colId / icComm->numCommColumns();
}

int HyPerCol::addLayer(HyPerLayer * l)
{
   assert((size_t) numLayers <= layerArraySize);

   // Check for duplicate layer names (currently breaks InitWeightsTest, so commented out)
   // for(int k=0; k<numLayers; k++) {
   //    if( !strcmp(l->getName(), layers[k]->getName())) {
   //       fprintf(stderr, "Error: Layers %d and %d have the same name \"%s\".\n", k, numLayers, l->getName());
   //       exit(EXIT_FAILURE);
   //    }
   // }

   if( (size_t) numLayers ==  layerArraySize ) {
      layerArraySize += RESIZE_ARRAY_INCR;
      HyPerLayer ** newLayers = (HyPerLayer **) malloc( layerArraySize * sizeof(HyPerLayer *) );
      assert(newLayers);
      for(int k=0; k<numLayers; k++) {
         newLayers[k] = layers[k];
      }
      free(layers);
      layers = newLayers;
   }
   l->columnWillAddLayer(icComm, numLayers);
   layers[numLayers++] = l;
   if (l->getPhase() >= numPhases) numPhases = l->getPhase()+1;
   return (numLayers - 1);
}

int HyPerCol::addConnection(HyPerConn * conn)
{
   int connId = numConnections;

   assert((size_t) numConnections <= connectionArraySize);
   // Check for duplicate connection names (currently breaks InitWeightsTest, so commented out)
   // for(int k=0; k<numConnections; k++) {
   //    if( !strcmp(conn->getName(), connections[k]->getName())) {
   //       fprintf(stderr, "Error: Layers %d and %d have the same name \"%s\".\n", k, numLayers, conn->getName());
   //       exit(EXIT_FAILURE);
   //    }
   // }
   if( (size_t) numConnections == connectionArraySize ) {
      connectionArraySize += RESIZE_ARRAY_INCR;
      HyPerConn ** newConnections = (HyPerConn **) malloc( connectionArraySize * sizeof(HyPerConn *) );
      assert(newConnections);
      for(int k=0; k<numConnections; k++) {
         newConnections[k] = connections[k];
      }
      free(connections);
      connections = newConnections;
   }

   // numConnections is the ID of this connection
   // subscribe call moved to HyPerCol::initPublishers, since it needs to be after the publishers are initialized.
   // icComm->subscribe(conn);

   connections[numConnections++] = conn;

   return connId;
}

int HyPerCol::run(long int nTimeSteps)
{
   if( checkMarginWidths() != PV_SUCCESS ) {
      fprintf(stderr, "Margin width failure; unable to continue.\n");
      return PV_MARGINWIDTH_FAILURE;
   }

   if( outputNamesOfLayersAndConns ) {
      assert( icComm->commRank() == 0 );
      printf("Dumping layer and connection names to \"%s\"\n", outputNamesOfLayersAndConns);
      PV_Stream * outputNamesStream = PV_fopen(outputNamesOfLayersAndConns,"w");
      if( outputNamesStream == NULL ) {
         fprintf(stderr, "HyPerCol \"%s\" unable to open \"%s\" for writing: error %d.  Exiting.\n", name, outputNamesOfLayersAndConns, errno);
         exit(errno);
      }
      fprintf(outputNamesStream->fp, "Layers and Connections in HyPerCol \"%s\"\n\n", name);
      for( int k=0; k<numLayers; k++ ) {
         fprintf(outputNamesStream->fp, "    Layer % 4d: %s\n", k, layers[k]->getName());
      }
      fprintf(outputNamesStream->fp, "\n");
      for( int k=0; k<numConnections; k++ ) {
         fprintf(outputNamesStream->fp, "    Conn. % 4d: %s\n", k, connections[k]->getName());
      }
      int fcloseStatus = PV_fclose(outputNamesStream);
      if( fcloseStatus != 0 ) {
         fprintf(stderr, "Warning: Attempting to close output file \"%s\" generated an error.\n", outputNamesOfLayersAndConns);
      }
      outputNamesStream = NULL;
   }

   stopTime = simTime + nTimeSteps * deltaTime;
   const bool exitOnFinish = false;

   if (!isInitialized) {
      initFinish();
   }

   initPublishers(); // create the publishers and their data stores

   numSteps = nTimeSteps;

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]: HyPerCol: running...\n");  fflush(stdout);
   }
#endif

   // Initialize either by loading from checkpoint, or calling initializeState
   // This needs to happen after initPublishers so that we can initialize the values in the data stores,
   // and before the layers' publish calls so that the data in border regions gets copied correctly.
   if ( checkpointReadFlag ) {
      int str_len = snprintf(NULL, 0, "%s/Checkpoint%ld", checkpointReadDir, cpReadDirIndex);
      size_t str_size = (size_t) (str_len+1);
      char * cpDir = (char *) malloc( str_size*sizeof(char) );
      snprintf(cpDir, str_size, "%s/Checkpoint%ld", checkpointReadDir, cpReadDirIndex);
      checkpointRead(cpDir);
      // Lines below commented out 2012-10-20.  We shouldn't delete the checkpoint we read from, for archival purposes
//      if (checkpointWriteFlag && deleteOlderCheckpoints) {
//         int chars_needed = snprintf(lastCheckpointDir, PV_PATH_MAX, "%s", cpDir);
//         if (chars_needed >= PV_PATH_MAX) {
//            if (icComm->commRank()==0) {
//               fprintf(stderr, "checkpointRead error: path \"%s\" is too long.\n", cpDir);
//            }
//            abort();
//         }
//      }
   }
   else {
      for ( int l=0; l<numLayers; l++ ) {
         layers[l]->initializeState();
      }
   }

   parameters()->warnUnread();

   // publish initial conditions
   //
   for (int l = 0; l < numLayers; l++) {
      layers[l]->publish(icComm, simTime);
   }

   // wait for all published data to arrive
   //
   for (int l = 0; l < numLayers; l++) {
      icComm->wait(layers[l]->getLayerId());
   }

   // output initial conditions
   for (int c = 0; c < numConnections; c++) {
      connections[c]->outputState(simTime);
   }
   for (int l = 0; l < numLayers; l++) {
      layers[l]->outputState(simTime);
   }

   if (runDelegate) {
      // let delegate advance the time
      //
      runDelegate->run(simTime, stopTime);
   }

#ifdef TIMER_ON
   start_clock();
#endif
   // time loop
   //
   long int step = 0;
   int status = PV_SUCCESS;
   while (simTime < stopTime && status != PV_EXIT_NORMALLY) {
      if( checkpointWriteFlag && advanceCPWriteTime() ) {
         if ( !checkpointReadFlag || strcmp(checkpointReadDir, checkpointWriteDir) || cpReadDirIndex!=currentStep ) {
            if (icComm->commRank()==0) {
               printf("Checkpointing, simTime = %f\n", simulationTime());
            }

            // Commented out Nov 4, 2012
            // if( currentStep >= HYPERCOL_DIRINDEX_MAX+1 ) {
            //    if( icComm->commRank() == 0 ) {
            //       fflush(stdout);
            //       fprintf(stderr, "Column \"%s\": step number exceeds maximum value %d.  Exiting\n", name, HYPERCOL_DIRINDEX_MAX);
            //    }
            //    exit(EXIT_FAILURE);
            // }
            char cpDir[PV_PATH_MAX];
            int chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Checkpoint%ld", checkpointWriteDir, currentStep);
            if(chars_printed >= PV_PATH_MAX) {
               if (icComm->commRank()==0) {
                  fprintf(stderr,"HyPerCol::run error.  Checkpoint directory \"%s/Checkpoint%ld\" is too long.\n", checkpointWriteDir, currentStep);
                  abort();
               }
            }
            checkpointWrite(cpDir);
         }
         else {
            if (icComm->commRank()==0) {
               printf("Skipping checkpoint at time %f, since this would clobber the checkpointRead checkpoint.\n", simulationTime());
            }
         }
      }
      status = advanceTime(simTime);

      step += 1;
#ifdef TIMER_ON
      if (step == 10) start_clock();
#endif

   }  // end time loop

#ifdef DEBUG_OUTPUT
   if (columnId() == 0) {
      printf("[0]: HyPerCol::run done...\n");  fflush(stdout);
   }
#endif

   exitRunLoop(exitOnFinish);

#ifdef TIMER_ON
   stop_clock();
#endif

   return PV_SUCCESS;
}

int HyPerCol::initPublishers() {
   for( int l=0; l<numLayers; l++ ) {
      PVLayer * clayer = layers[l]->getCLayer();
      icComm->addPublisher(layers[l], clayer->activity->numItems, clayer->numDelayLevels);
   }
   for( int c=0; c<numConnections; c++ ) {
      icComm->subscribe(connections[c]);
   }

   return PV_SUCCESS;
}

int HyPerCol::advanceTime(double sim_time)
{
#ifdef TIMESTEP_OUTPUT
   if (currentStep%progressStep == 0 && columnId() == 0) {
      if (writeProgressToErr) {
         fprintf(stderr, "   [%d]: time==%f\n", columnId(), sim_time);
      }
      else
      {
         printf("   [%d]: time==%f\n", columnId(), sim_time);
      }
   }
#endif

   runTimer->start();

   // At this point all activity from the previous time step has
   // been delivered to the data store.
   //

   int status = PV_SUCCESS;
   bool exitAfterUpdate = false;

   // update the connections (weights)
   //
   for (int c = 0; c < numConnections; c++) {
      status = connections[c]->updateState(sim_time, deltaTime);
      if (!exitAfterUpdate) {
		  exitAfterUpdate = status == PV_EXIT_NORMALLY;
      }
   }
   for (int c = 0; c < numConnections; c++) {
      connections[c]->outputState(sim_time);
   }

   // Each layer's phase establishes a priority for updating
   for (int phase=0; phase<numPhases; phase++) {

      // clear GSyn buffers
      for(int l = 0; l < numLayers; l++) {
         if (layers[l]->getPhase() != phase) continue;
         layers[l]->resetGSynBuffers(sim_time, deltaTime);
         layers[l]->recvAllSynapticInput();
      }
      //    for (int l = 0; l < numLayers; l++) {
      //       // deliver new synaptic activity to any
      //       // postsynaptic layers for which this
      //       // layer is presynaptic.
      //       layers[l]->triggerReceive(icComm);
      //    }

      // Update the layers (activity)
      // We don't put updateState in the same loop over layers as recvAllSynapticInput
      // because we plan to have updateState update the datastore directly, and
      // recvSynapticInput uses the datastore to compute GSyn.
      for(int l = 0; l < numLayers; l++) {
         if (layers[l]->getPhase() != phase) continue;
         status = layers[l]->updateState(sim_time, deltaTime);
		 if (!exitAfterUpdate) {
			 exitAfterUpdate = status == PV_EXIT_NORMALLY;
		 }
      }

      // This loop separate from the update layer loop above
      // to provide time for layer data to be copied from
      // the OpenCL device.
      //
      for (int l = 0; l < numLayers; l++) {
         if (layers[l]->getPhase() != phase) continue;
         // after updateBorder completes all necessary data has been
         // copied from the device (GPU) to the host (CPU)
         layers[l]->updateBorder(sim_time, deltaTime); // TODO rename updateBorder?

         // TODO - move this to layer
         // Advance time level so we have a new place in data store
         // to copy the data.  This should be done immediately before
         // publish so there is a place to publish and deliver the data to.
         // No one can access the data store (except to publish) until
         // wait has been called.  This should be fixed so that publish goes
         // to last time level and level is advanced only after wait.
         icComm->increaseTimeLevel(layers[l]->getLayerId());

         layers[l]->publish(icComm, sim_time);
         //    }
         //
         //    // wait for all published data to arrive
         //    //
         //    for (int l = 0; l < numLayers; l++) {
         layers[l]->waitOnPublish(icComm);
         //    }
         //
         //    // also calls layer probes
         //    for (int l = 0; l < numLayers; l++) {
         layers[l]->outputState(sim_time);
      }

   }

   // make sure simTime is updated even if HyPerCol isn't running time loop

   double outputTime = simTime; // so that outputState is called with the correct time
                               // but doesn't effect runTimer

   simTime = sim_time + deltaTime;
   currentStep++;

   runTimer->stop();

   outputState(outputTime);

   if (exitAfterUpdate) {
	   status = PV_EXIT_NORMALLY;
   }

   return status;
}

bool HyPerCol::advanceCPWriteTime() {
   // returns true if nextCPWrite{Step,Time} has been advanced
   bool advanceCPTime;
   if( cpWriteStepInterval>0 ) {
      assert(cpWriteTimeInterval<0.0);
      advanceCPTime = currentStep >= nextCPWriteStep;
      if( advanceCPTime ) {
         nextCPWriteStep += cpWriteStepInterval;
      }
   }
   else if( cpWriteTimeInterval>0.0) {
      assert(cpWriteStepInterval<0);
      advanceCPTime = simTime >= nextCPWriteTime;
      if( advanceCPTime ) {
         nextCPWriteTime += cpWriteTimeInterval;
      }
   }
   else {
      assert( false ); // routine should only be called if one of cpWrite{Step,Time}Interval is positive
      advanceCPTime = false;
   }
   return advanceCPTime;
}

int HyPerCol::checkpointRead(const char * cpDir) {
   struct timestamp_struct {
      double time; // time measured in units of dt
      long int step; // step number, usually time/dt
   };
   struct timestamp_struct timestamp;
   size_t timestamp_size = sizeof(struct timestamp_struct);
   assert(sizeof(struct timestamp_struct) == sizeof(long int) + sizeof(double));
   if( icCommunicator()->commRank()==0 ) {
      char timestamppath[PV_PATH_MAX];
      int chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.bin", cpDir);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "HyPerCol::checkpointRead error: path \"%s/timeinfo.bin\" is too long.\n", cpDir);
         abort();
      }
      PV_Stream * timestampfile = PV_fopen(timestamppath,"r");
      if (timestampfile == NULL) {
         fprintf(stderr, "HyPerCol::checkpointRead error: unable to open \"%s\" for reading.\n", timestamppath);
         abort();
      }
      long int startpos = getPV_StreamFilepos(timestampfile);
      PV_fread(&timestamp,1,timestamp_size,timestampfile);
      long int endpos = getPV_StreamFilepos(timestampfile);
      assert(endpos-startpos==(int)timestamp_size);
      PV_fclose(timestampfile);
   }
#ifdef PV_USE_MPI
   MPI_Bcast(&timestamp,(int) timestamp_size,MPI_CHAR,0,icCommunicator()->communicator());
#endif // PV_USE_MPI
   simTime = timestamp.time;
   currentStep = timestamp.step;
   double checkTime;
   for( int l=0; l<numLayers; l++ ) {
      layers[l]->checkpointRead(cpDir, &checkTime);
      assert(checkTime==simTime);
   }
   for( int c=0; c<numConnections; c++ ) {
      connections[c]->checkpointRead(cpDir, &checkTime);
      assert(checkTime==simTime);
   }
   if(checkpointWriteFlag) {
      if( cpWriteStepInterval > 0) {
         assert(cpWriteTimeInterval<0.0f);
         nextCPWriteStep = currentStep; // checkpointWrite should be called before any timesteps,
             // analogous to checkpointWrite being called immediately after initialization on a fresh run.
      }
      else if( cpWriteTimeInterval > 0.0f ) {
         assert(cpWriteStepInterval<0);
         nextCPWriteTime = simTime; // checkpointWrite should be called before any timesteps
      }
      else {
         assert(false); // if checkpointWriteFlag is set, one of cpWrite{Step,Time}Interval should be positive
      }
   }
   return PV_SUCCESS;
}

int HyPerCol::checkpointWrite(const char * cpDir) {
   if (icCommunicator()->commRank()==0) {
      printf("Checkpointing to directory \"%s\" at simTime = %f\n", cpDir, simTime);
   }

   // Commented out Nov 4, 2012.
   // if( currentStep >= HYPERCOL_DIRINDEX_MAX+1 ) {
   //    if( icComm->commRank() == 0 ) {
   //       fflush(stdout);
   //       fprintf(stderr, "Column \"%s\": step number exceeds maximum value %d.  Exiting\n", name, HYPERCOL_DIRINDEX_MAX);
   //   }
   //    exit(EXIT_FAILURE);
   // }

   if (columnId()==0) {
      struct stat timeinfostat;
      char timeinfofilename[PV_PATH_MAX];
      int chars_needed = snprintf(timeinfofilename, PV_PATH_MAX, "%s/timeinfo.bin", cpDir);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "HyPerCol::checkpointRead error: path \"%s/timeinfo.bin\" is too long.\n", cpDir);
         abort();
      }
      int statstatus = stat(timeinfofilename, &timeinfostat);
      if (statstatus == 0) {
         fprintf(stderr, "Warning: Checkpoint directory \"%s\" has existing timeinfo.bin, which is now being deleted.\n", timeinfofilename);
         int unlinkstatus = unlink(timeinfofilename);
         if (unlinkstatus != 0) {
            fprintf(stderr, "Error deleting \"%s\": %s\n", timeinfofilename, strerror(errno));
            abort();
         }
      }
   }

   ensureDirExists(cpDir);
   for( int l=0; l<numLayers; l++ ) {
      layers[l]->checkpointWrite(cpDir);
   }
   for( int c=0; c<numConnections; c++ ) {
      connections[c]->checkpointWrite(cpDir);
   }

   // Note: timeinfo should be done at the end of the checkpointing, so that its presence serves as a flag that the checkpoint has completed.
   if( icCommunicator()->commRank()==0 ) {
      char timestamppath[PV_PATH_MAX];
      int chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.bin", cpDir);
      assert(chars_needed < PV_PATH_MAX);
      PV_Stream * timestampfile = PV_fopen(timestamppath,"w");
      assert(timestampfile);
      PV_fwrite(&simTime,1,sizeof(double),timestampfile);
      PV_fwrite(&currentStep,1,sizeof(long int),timestampfile);
      PV_fclose(timestampfile);
      chars_needed = snprintf(timestamppath, PV_PATH_MAX, "%s/timeinfo.txt", cpDir);
      assert(chars_needed < PV_PATH_MAX);
      timestampfile = PV_fopen(timestamppath,"w");
      assert(timestampfile);
      fprintf(timestampfile->fp,"time = %g\n", simTime);
      fprintf(timestampfile->fp,"timestep = %ld\n", currentStep);
      PV_fclose(timestampfile);
   }

   if (deleteOlderCheckpoints) {
      assert(checkpointWriteFlag); // checkpointWrite is called by exitRunLoop when checkpointWriteFlag is false; in this case deleteOlderCheckpoints should be false as well.
      if (lastCheckpointDir[0]) {
         if (icComm->commRank()==0) {
            struct stat lcp_stat;
            int statstatus = stat(lastCheckpointDir, &lcp_stat);
            if ( statstatus!=0 || !(lcp_stat.st_mode & S_IFDIR) ) {
               if (statstatus==0) {
                  fprintf(stderr, "Error deleting older checkpoint: failed to stat \"%s\": %s.\n", lastCheckpointDir, strerror(errno));
               }
               else {
                  fprintf(stderr, "Deleting older checkpoint: \"%s\" exists but is not a directory.\n", lastCheckpointDir);
               }
            }
#define RMRFSIZE (PV_PATH_MAX + 13)
            char rmrf_string[RMRFSIZE];
            int chars_needed = snprintf(rmrf_string, RMRFSIZE, "rm -r '%s'", lastCheckpointDir);
            assert(chars_needed < RMRFSIZE);
#undef RMRFSIZE
            system(rmrf_string);
         }
      }
      int chars_needed = snprintf(lastCheckpointDir, PV_PATH_MAX, "%s", cpDir);
      assert(chars_needed < PV_PATH_MAX);
   }

   if (icComm->commRank()==0) {
      fprintf(stderr, "checkpointWrite complete. simTime = %f\n", simTime);
   }
   return PV_SUCCESS;
}

int HyPerCol::outputParams(const char * filename) {
   int status = PV_SUCCESS;
#ifdef PV_USE_MPI
   int rank=icComm->commRank();
#else
   int rank=0;
#endif
   if( rank==0 && filename != NULL && filename[0] != '\0' ) {
      char printParamsPath[PV_PATH_MAX];
      int len;
      if (filename[0] == '/') { // filename is absolute path
         len = snprintf(printParamsPath, PV_PATH_MAX, "%s", filename);
      }
      else { // filename is relative path from outputPath
         len = snprintf(printParamsPath, PV_PATH_MAX, "%s/%s", outputPath, filename);
      }
      if( len < PV_PATH_MAX ) {
         PV_Stream * pvstream = PV_fopen(printParamsPath, "w");
         if( pvstream != NULL ) {
            status = params->outputParams(pvstream->fp);
            if( status != PV_SUCCESS ) {
               fprintf(stderr, "outputParams: Error copying params to \"%s\"\n", printParamsPath);
            }
            PV_fclose(pvstream); pvstream = NULL;
         }
         else {
            status = errno;
            fprintf(stderr, "outputParams error opening \"%s\" for writing: %s\n", printParamsPath, strerror(errno));
         }
      }
      else {
         fprintf(stderr, "outputParams: ");
         if (filename[0] != '/') fprintf(stderr, "outputPath + ");
         fprintf(stderr, "printParamsFilename gives too long a filename.  Parameters will not be printed.\n");
      }
   }
   return status;
}

#ifdef UNDERCONSTRUCTION // The plan is to output all params, whether they were set in the params file or not.
int HyPerCol::outputParamsXML(const char * filename) {
   int status = PV_SUCCESS;
#ifdef PV_USE_MPI
   int rank=icComm->commRank();
#else
   int rank=0;
#endif
   if( rank==0 && filename != NULL && filename[0] != '\0' ) {
      char printParamsPath[PV_PATH_MAX];
      int len;
      if (filename[0] == '/') { // filename is absolute path
         len = snprintf(printParamsPath, PV_PATH_MAX, "%s", filename);
      }
      else { // filename is relative path from outputPath
         len = snprintf(printParamsPath, PV_PATH_MAX, "%s/%s", outputPath, filename);
      }
      if( len < PV_PATH_MAX ) {
         PV_Stream * pvstream = PV_fopen(printParamsPath, "w");
         if( pvstream != NULL ) {
            status = outputParamsXML(pvstream);
            if( status != PV_SUCCESS ) {
               fprintf(stderr, "outputParamsXML: Error copying params to \"%s\"\n", printParamsPath);
            }
            PV_fclose(pvstream); pvstream = NULL;
         }
         else {
            status = errno;
            fprintf(stderr, "outputParamsXML error opening \"%s\" for writing: %s\n", printParamsPath, strerror(errno));
         }
      }
      else {
         fprintf(stderr, "outputParams: ");
         if (filename[0] != '/') fprintf(stderr, "outputPath + ");
         fprintf(stderr, "paramsXMLFilename gives too long a filename.  Parameters will not be printed.\n");
      }
   }
   return status;
}

int HyPerCol::outputParamsXML(PV_Stream * pvstream) {
   assert(pvstream!=NULL && pvstream->fp!=NULL);
   FILE * fp = pvstream->fp;
   fprintf(fp, "<?xml version='1.0' encoding=\"UTF-8\"?>\n");
   fprintf(fp, "<params>\n");
   int indentation=1;
   outputParamGroup(pvstream, "HyPerCol", name, indentation);
   indentation++;
   outputParamInt(pvstream, "nx", nxGlobal, indentation);
   outputParamInt(pvstream, "ny", nyGlobal, indentation);
   outputParamDouble(pvstream, "dt", deltaTime, indentation);
   outputParamUnsignedLongInt(pvstream, "randomSeed", random_seed, indentation);
   outputParamLongInt(pvstream, "numSteps", numSteps, indentation);
   outputParamLongInt(pvstream, "progressStep", progressStep, indentation);
   outputParamBoolean(pvstream, "writeProgressToErr", writeProgressToErr, indentation);
   outputParamFilename(pvstream, "outputPath", outputPath, indentation);
   outputParamFilename(pvstream, "paramsXMLFilename", pvstream->name, indentation);
   outputParamInt(pvstream, "filenamesContainLayerNames", filenamesContainLayerNames, indentation);
   outputParamInt(pvstream, "filenamesContainConnectionNames", filenamesContainConnectionNames, indentation);
   outputParamBoolean(pvstream, "checkpointRead", checkpointReadFlag, indentation);
   outputParamFilename(pvstream, "checkpointReadDir", checkpointReadDir, indentation);
   outputParamLongInt(pvstream, "checkpointReadDirIndex", cpReadDirIndex, indentation);
   outputParamBoolean(pvstream, "checkpointWrite", checkpointWriteFlag, indentation);
   outputParamFilename(pvstream, "checkpointWriteDir", checkpointWriteDir, indentation);
   outputParamLongInt(pvstream, "checkpointWriteStepInterval", cpWriteStepInterval, indentation);
   outputParamDouble(pvstream, "chekpointWriteTimeInterval", cpWriteTimeInterval, indentation);
   outputParamBoolean(pvstream, "deleteOlderCheckpoints", deleteOlderCheckpoints, indentation);
   outputParamBoolean(pvstream, "suppressLastOutput", suppressLastOutput, indentation);
   indentation--;
   for (int l=0; l<numLayers; l++) {
      // layers[l]->outputParamsXML(pvstream); // Need to add to HyPerLayer
   }
   for (int c=0; c<numConnections; c++) {
      // connections[l]->outputParamsXML(pvstream); // Need to add to HyPerConnection
   }
   outputParamCloseGroup(pvstream, "HyPerCol", indentation);
   fprintf(fp, "</params>\n");
   return PV_SUCCESS;
}

int HyPerCol::outputParamGroup(PV_Stream * pvstream, const char * classname, const char * groupname, int indentation) {
   indent(pvstream, indentation);
   fprintf(pvstream->fp, "<%s name=\"%s\">\n", classname, groupname);
   return PV_SUCCESS;
}

int HyPerCol::outputParamCloseGroup(PV_Stream * pvstream, const char * classname, int indentation) {
   indent(pvstream, indentation);
   fprintf(pvstream->fp, "</%s>\n", classname);
   return PV_SUCCESS;
}

int HyPerCol::outputParamInt(PV_Stream * pvstream, const char * paramname, int value, int indentation) {
   indent(pvstream, indentation);
   fprintf(pvstream->fp, "<param name=\"%s\" type=\"int\">%d</param>\n", paramname, value);
   return PV_SUCCESS;
}

int HyPerCol::outputParamLongInt(PV_Stream * pvstream, const char * paramname, long int value, int indentation) {
   indent(pvstream, indentation);
   fprintf(pvstream->fp, "<param name=\"%s\" type=\"long int\">%ld</param>\n", paramname, value);
   return PV_SUCCESS;
}

int HyPerCol::outputParamUnsignedLongInt(PV_Stream * pvstream, const char * paramname, unsigned long int value, int indentation) {
   indent(pvstream, indentation);
   fprintf(pvstream->fp, "<param name=\"%s\" type=\"unsigned long int\">%lu</param>\n", paramname, value);
   return PV_SUCCESS;
}

int HyPerCol::outputParamFloat(PV_Stream * pvstream, const char * paramname, float value, int indentation) {
   indent(pvstream, indentation);
   fprintf(pvstream->fp, "<param name=\"%s\" type=\"float\">%f (", paramname, value);
   hexdump(pvstream, value);
   fprintf(pvstream->fp, ")</param>\n");
   return PV_SUCCESS;
}

int HyPerCol::outputParamDouble(PV_Stream * pvstream, const char * paramname, double value, int indentation) {
   indent(pvstream, indentation);
   fprintf(pvstream->fp, "<param name=\"%s\" type=\"double\">%f (", paramname, value);
   hexdump(pvstream, value);
   fprintf(pvstream->fp, ")</param>\n");
   return PV_SUCCESS;
}

template <typename T> int HyPerCol::hexdump(PV_Stream * pvstream, T value) {
   size_t sz = sizeof(value);
   unsigned char c[sz];
   memcpy(c, &value, sz);
   fprintf(pvstream->fp, "0x");
   for (size_t j=sz; j>0;) {
      fprintf(pvstream->fp, "%02x", c[--j]);
   }
   return PV_SUCCESS;
}
template int HyPerCol::hexdump<float>(PV_Stream * pvstream, float value);
template int HyPerCol::hexdump<double>(PV_Stream * pvstream, double value);

int HyPerCol::outputParamBoolean(PV_Stream * pvstream, const char * paramname, bool value, int indentation) {
   indent(pvstream, indentation);
   const char * truestring = "true";
   const char * falsestring = "false";
   fprintf(pvstream->fp, "<param name=\"%s\" type=\"boolean\">%s</param>\n", paramname, value?truestring:falsestring);
   return PV_SUCCESS;
}

int HyPerCol::outputParamFilename(PV_Stream * pvstream, const char * paramname, const char * value, int indentation) {
   indent(pvstream, indentation);
   fprintf(pvstream->fp, "<param name=\"%s\" type=\"filename\">\"", paramname);
   if (value) fprintf(pvstream->fp, "%s", value);
   fprintf(pvstream->fp, "\"</param>\n");
   return PV_SUCCESS;
}

int HyPerCol::outputParamString(PV_Stream * pvstream, const char * paramname, const char * value, int indentation) {
   indent(pvstream, indentation);
   fprintf(pvstream->fp, "<param name=\"%s\" type=\"filename\">\"", paramname);
   if (value) fprintf(pvstream->fp, "%s", value);
   fprintf(pvstream->fp, "\"</param>\n");
   return PV_SUCCESS;
}

int HyPerCol::indent(PV_Stream * pvstream, int indentation) {
   const char indentstring[] = "   ";
   int printed = 0;
   for (int k=0; k<indentation; k++) {
      int fprintstatus = fprintf(pvstream->fp, indentstring);
      assert(fprintstatus = strlen(indentstring));
      printed += fprintstatus;
   }
   assert(indentation<0 || printed==indentation*(int)strlen(indentstring));
   return printed;
}
#endif // UNDERCONSTRUCTION

int HyPerCol::exitRunLoop(bool exitOnFinish)
{
   int status = 0;

   // output final state of layers and connections
   //

   char cpDir[PV_PATH_MAX];
   if (checkpointWriteFlag || !suppressLastOutput) {
      int chars_printed;
      if (checkpointWriteFlag) {
         chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Checkpoint%ld", checkpointWriteDir, currentStep);
      }
      else {
         assert(!suppressLastOutput);
         chars_printed = snprintf(cpDir, PV_PATH_MAX, "%s/Last", outputPath);
      }
      if(chars_printed >= PV_PATH_MAX) {
         if (icComm->commRank()==0) {
            fprintf(stderr,"HyPerCol::run error.  Checkpoint directory \"%s/Checkpoint%ld\" is too long.\n", checkpointWriteDir, currentStep);
            abort();
         }
      }
      checkpointWrite(cpDir);
   }

#ifdef OBSOLETE // Marked obsolete July 13, 2012.  Final output is written to {outputPath}/Last, above, using CheckpointWrite
   bool last = true;
   for (int l = 0; l < numLayers; l++) {
      layers[l]->writeState(simTime, last);
   }

   for (int c = 0; c < numConnections; c++) {
      connections[c]->outputState(simTime, last);
   }
#endif // OBSOLETE

   if (exitOnFinish) {
      delete this;
      exit(0);
   }

   return status;
}

int HyPerCol::initializeThreads(int device)
{
   clDevice = new CLDevice(device);
   return 0;
}

#ifdef PV_USE_OPENCL
int HyPerCol::finalizeThreads()
{
   delete clDevice;
   return 0;
}
#endif // PV_USE_OPENCL

int HyPerCol::loadState()
{
   return 0;
}

#ifdef OBSOLETE // Marked obsolete Nov 1, 2011.  Nobody calls this routine and it will be supplanted by checkpointWrite()
int HyPerCol::writeState()
{
   for (int l = 0; l < numLayers; l++) {
      layers[l]->writeState(simTime);
   }
   return 0;
}
#endif // OBSOLETE


int HyPerCol::insertProbe(ColProbe * p)
{
   ColProbe ** newprobes;
   newprobes = (ColProbe **) malloc( ((size_t) (numProbes + 1)) * sizeof(ColProbe *) );
   assert(newprobes != NULL);

   for (int i = 0; i < numProbes; i++) {
      newprobes[i] = probes[i];
   }
   delete probes;

   probes = newprobes;
   probes[numProbes] = p;

   return ++numProbes;
}

int HyPerCol::outputState(double time)
{
   for( int n = 0; n < numProbes; n++ ) {
       probes[n]->outputState(time, this);
   }
   return PV_SUCCESS;
}


HyPerLayer * HyPerCol::getLayerFromName(const char * layerName) {
   int n = numberOfLayers();
   for( int i=0; i<n; i++ ) {
      HyPerLayer * curLayer = getLayer(i);
      assert(curLayer);
      const char * curLayerName = curLayer->getName();
      assert(curLayerName);
      if( !strcmp( curLayer->getName(), layerName) ) return curLayer;
   }
   return NULL;
}

HyPerConn * HyPerCol::getConnFromName(const char * connName) {
   if( connName == NULL ) return NULL;
   int n = numberOfConnections();
   for( int i=0; i<n; i++ ) {
      HyPerConn * curConn = getConnection(i);
      assert(curConn);
      const char * curConnName = curConn->getName();
      assert(curConnName);
      if( !strcmp( curConn->getName(), connName) ) return curConn;
   }
   return NULL;
}

unsigned long HyPerCol::getRandomSeed() {
   unsigned long t = 0UL;
   int rootproc = 0;
   if (columnId()==rootproc) {
       t = time((time_t *) NULL);
   }
   MPI_Bcast(&t, 1, MPI_UNSIGNED_LONG, rootproc, icComm->communicator());
   return t;
}

int HyPerCol::checkMarginWidths() {
   // For each connection, make sure that the pre-synaptic margin width is
   // large enough for the patch size.

   // TODO instead of having marginWidth supplied to HyPerLayers in the
   // params.pv file, calculate them based on the patch sizes here.
   // Hard part:  numExtended-sized quantities (e.g. clayer->activity) can't
   // be allocated and initialized until after nPad is determined.

   int status = PV_SUCCESS;
   int status1, status2;
   for( int c=0; c < numConnections; c++ ) {
      HyPerConn * conn = connections[c];
      HyPerLayer * pre = conn->preSynapticLayer();
      HyPerLayer * post = conn->postSynapticLayer();

      int xScalePre = pre->getXScale();
      int xScalePost = post->getXScale();
      status1 = zCheckMarginWidth(conn, "x", conn->xPatchSize(), xScalePre, xScalePost, status);

      int yScalePre = pre->getYScale();
      int yScalePost = post->getYScale();
      status2 = zCheckMarginWidth(conn, "y", conn->yPatchSize(), yScalePre, yScalePost, status1);
      status = (status == PV_SUCCESS && status1 == PV_SUCCESS && status2 == PV_SUCCESS) ?
               PV_SUCCESS : PV_MARGINWIDTH_FAILURE;
   }
   for( int l=0; l < numLayers; l++ ) {
      HyPerLayer * layer = layers[l];
      status1 = lCheckMarginWidth(layer, "x", layer->getLayerLoc()->nx, layer->getLayerLoc()->nxGlobal, status);
      status2 = lCheckMarginWidth(layer, "y", layer->getLayerLoc()->ny, layer->getLayerLoc()->nyGlobal, status1);
      status = (status == PV_SUCCESS && status1 == PV_SUCCESS && status2 == PV_SUCCESS) ?
               PV_SUCCESS : PV_MARGINWIDTH_FAILURE;
   }
   return status;
}  // end HyPerCol::checkMarginWidths()

int HyPerCol::zCheckMarginWidth(HyPerConn * conn, const char * dim, int patchSize, int scalePre, int scalePost, int prevStatus) {
   int status;
   int scaleDiff = scalePre - scalePost;
   // if post has higher neuronal density than pre, scaleDiff < 0.
   HyPerLayer * pre = conn->preSynapticLayer();
   int padding = conn->preSynapticLayer()->getLayerLoc()->nb;
   int needed = scaleDiff > 0 ? ( patchSize/( (int) pow(2,scaleDiff) )/2 ) :
                                ( (patchSize/2) * ( (int) pow(2,-scaleDiff) ) );
   if( padding < needed ) {
      if( prevStatus == PV_SUCCESS ) {
         fprintf(stderr, "Margin width error.\n");
      }
      fprintf(stderr, "Connection \"%s\", dimension %s:\n", conn->getName(), dim);
      fprintf(stderr, "    Pre-synaptic margin width %d, patch size %d, presynaptic scale %d, postsynaptic scale %d\n",
              padding, patchSize, scalePre, scalePost);
      fprintf(stderr, "    Layer %s needs margin width of at least %d\n", pre->getName(), needed);
      if( numberOfColumns() > 1 || padding > 0 ) {
         status = PV_MARGINWIDTH_FAILURE;
      }
      else {
         fprintf(stderr, "Continuing, but there may be undesirable edge effects.\n");
         status = PV_SUCCESS;
      }
   }
   else status = PV_SUCCESS;
   return status;
}

int HyPerCol::lCheckMarginWidth(HyPerLayer * layer, const char * dim, int layerSize, int layerGlobalSize, int prevStatus) {
   int status;
   int nb = layer->getLayerLoc()->nb;
   if( layerSize < nb) {
      if( prevStatus == PV_SUCCESS ) {
         fprintf(stderr, "Margin width error.\n");
      }
      fprintf(stderr, "Layer \"%s\", dimension %s:\n", layer->getName(), dim);
      fprintf(stderr, "    Pre-synaptic margin width %d, overall layer size %d, layer size per process %d\n", nb, layerGlobalSize, layerSize);
      fprintf(stderr, "    Use either fewer processes in dimension %s, or a margin size <= %d.\n", dim, layerSize);
      status = PV_MARGINWIDTH_FAILURE;
   }
   else status = PV_SUCCESS;
   return status;
}

template <typename T>
int HyPerCol::writeScalarToFile(const char * cp_dir, const char * group_name, const char * val_name, T val) {
   int status = PV_SUCCESS;
   if (columnId()==0)  {
      char filename[PV_PATH_MAX];
      int chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.bin", cp_dir, group_name, val_name);
      if (chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "writeScalarToFile error: path %s/%s_%s.bin is too long.\n", cp_dir, group_name, val_name);
         abort();
      }
      PV_Stream * pvstream = PV_fopen(filename, "w");
      if (pvstream==NULL) {
         fprintf(stderr, "writeScalarToFile error: unable to open path %s for writing.\n", filename);
         abort();
      }
      int num_written = PV_fwrite(&val, sizeof(val), 1, pvstream);
      if (num_written != 1) {
         fprintf(stderr, "writeScalarToFile error while writing to %s.\n", filename);
         abort();
      }
      PV_fclose(pvstream);
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.txt", cp_dir, group_name, val_name);
      assert(chars_needed < PV_PATH_MAX);
      std::ofstream fs;
      fs.open(filename);
      if (!fs) {
         fprintf(stderr, "writeScalarToFile error: unable to open path %s for writing.\n", filename);
         abort();
      }
      fs << val;
      fs << std::endl; // Can write as fs << val << std::endl, but eclipse flags that as an error 'Invalid overload of std::endl'
      fs.close();
   }
   return status;
}
// Declare the instantiations of writeScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
template int HyPerCol::writeScalarToFile<int>(char const * cpDir, const char * group_name, char const * val_name, int val);
template int HyPerCol::writeScalarToFile<long>(char const * cpDir, const char * group_name, char const * val_name, long val);
template int HyPerCol::writeScalarToFile<float>(char const * cpDir, const char * group_name, char const * val_name, float val);
template int HyPerCol::writeScalarToFile<double>(char const * cpDir, const char * group_name, char const * val_name, double val);

template <typename T>
int HyPerCol::readScalarFromFile(const char * cp_dir, const char * group_name, const char * val_name, T * val, T default_value) {
   int status = PV_SUCCESS;
   if( columnId() == 0 ) {
      char filename[PV_PATH_MAX];
      int chars_needed;
      chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_%s.bin", cp_dir, group_name, val_name);
      if(chars_needed >= PV_PATH_MAX) {
         fprintf(stderr, "HyPerLayer::readScalarFloat error: path %s/%s_%s.bin is too long.\n", cp_dir, group_name, val_name);
         abort();
      }
      PV_Stream * pvstream = PV_fopen(filename, "r");
      *val = default_value;
      if (pvstream==NULL) {
         std::cerr << "readScalarFromFile warning: unable to open path \"" << filename << "\" for reading.  Value used will be " << *val;
         std::cerr << std::endl;
         // fprintf(stderr, "HyPerLayer::readScalarFloat warning: unable to open path %s for reading.  value used will be %f\n", filename, default_value);
      }
      else {
         int num_read = PV_fread(val, sizeof(T), 1, pvstream);
         if (num_read != 1) {
            std::cerr << "readScalarFromFile warning: unable to read from \"" << filename << "\".  Value used will be " << *val;
            std::cerr << std::endl;
            // fprintf(stderr, "HyPerLayer::readScalarFloat warning: unable to read from %s.  value used will be %f\n", filename, default_value);
         }
         PV_fclose(pvstream);
      }
   }
#ifdef PV_USE_MPI
   MPI_Bcast(val, sizeof(T), MPI_CHAR, 0, icCommunicator()->communicator());
#endif // PV_USE_MPI

   return status;
}
// Declare the instantiations of readScalarToFile that occur in other .cpp files; otherwise you'll get linker errors.
template int HyPerCol::readScalarFromFile<int>(char const * cpDir, const char * group_name, char const * val_name, int * val, int default_value);
template int HyPerCol::readScalarFromFile<long>(char const * cpDir, const char * group_name, char const * val_name, long * val, long default_value);
template int HyPerCol::readScalarFromFile<float>(char const * cpDir, const char * group_name, char const * val_name, float * val, float default_value);
template int HyPerCol::readScalarFromFile<double>(char const * cpDir, const char * group_name, char const * val_name, double * val, double default_value);

} // PV namespace
