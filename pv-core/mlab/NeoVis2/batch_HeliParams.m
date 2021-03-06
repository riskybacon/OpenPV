
%% begin definition of the most volitile parameters
%% clip_name stores the directories that contain the individual frames
FLAVOR_ID = "Training"; %%   "Formative"; %%   "Challenge"; %% 
disp(["FLAVOR_ID = ", FLAVOR_ID]);
<<<<<<< HEAD
clip_flag = true; %% false; %%   
mask_flag = ~clip_flag; %% false; %% true; %% 
target_mask_flag = true && mask_flag; %% false; %% 
disp(["target_mask_flag = ", num2str(target_mask_flag)]);
object_type = {"Car"}; %%_distractor"}; %% 
disp(["object_type = ", object_type{1}]);
pvp_num_ODD_kernels = 1; %%
disp(["num_ODD_kernels = ", num2str(pvp_num_ODD_kernels)]);
if clip_flag
  clip_ids = [1:50]; %%  [26:26]; %%[1:25]; %% [7:17,21:22,30:31];
  clip_name = cell(length(clip_ids),1);
  for i_clip = 1 : length(clip_name)
    clip_name{i_clip} = num2str(clip_ids(i_clip), "%3.3i");
  endfor
else
  clip_name = cell(1);
  clip_name{1} = "mask"; %% "026"; %% 
endif
disp(["clip_name{1} = ", clip_name{1}]);
distractor_mask_flag = ~target_mask_flag && mask_flag;
disp(["distractor_mask_flag = ", num2str(distractor_mask_flag)]);
if clip_flag
  version_ids = [1]; %%[1:16]; %%  
else
  version_ids = [1]; %% [1:16]; %% 
endif
disp(["version_ids = ", mat2str(version_ids)]);
pvp_frame_size = [1080 1920]; %%  [256, 256]; %%
disp(["frame_size = ", mat2str(pvp_frame_size)]);
pvp_edge_type = "canny"; 
pvp_clique_id = "3way"; %% ""; %% 
%% end definition of the most volitile parameters

%% version_str stores the training or testing run index 
num_versions = length(version_ids);
if num_versions > 0
  version_str = cell(num_versions,1);
  for i_version = 1 : num_versions
    version_str{i_version} = num2str(version_ids(i_version), "%3.3i");   
  endfor
else
  num_versions = 1;
  version_str = cell(num_versions,1);
  version_str{1}="";
endif

global PVP_VERBOSE_FLAG
PVP_VERBOSE_FLAG = 0;

global pvp_home_path
global pvp_workspace_path
global pvp_mlab_path
global pvp_clique_path
pvp_home_path = ...
    [filesep, "home", filesep, "gkenyon", filesep];
pvp_workspace_path = ...
    [pvp_home_path, "workspace-sync-canto2", filesep];
pvp_mlab_path = ...
    [pvp_workspace_path, "PetaVision", filesep, "mlab", filesep];
pvp_clique_path = ...
    [pvp_workspace_path, "SynthCog3", filesep];


DATASET_ID = "Heli"; %% "amoeba"; %%"Tower"; %% "Tailwind"; %% 
dataset_id = tolower(DATASET_ID); %% 
flavor_id = tolower(FLAVOR_ID); %% 
pvp_repo_path = ...
    [filesep, "nh", filesep, "compneuro", filesep, "Data", filesep, "repo", filesep];
=======
clip_flag = false; %% true; %%  
mask_flag = ~clip_flag; %% false; %% true; %% 
target_mask_flag = true && mask_flag; %% false; %% 
disp(["target_mask_flag = ", num2str(target_mask_flag)]);
object_type = {"Car"}; %%_distractor"}; %% 
disp(["object_type = ", object_type{1}]);
pvp_num_ODD_kernels = 2; %%
disp(["num_ODD_kernels = ", num2str(pvp_num_ODD_kernels)]);
if clip_flag
  clip_ids = [26:26]; %%[1:25]; %% [1:50]; %%  [7:17,21:22,30:31];
  clip_name = cell(length(clip_ids),1);
  for i_clip = 1 : length(clip_name)
    clip_name{i_clip} = num2str(clip_ids(i_clip), "%3.3i");
  endfor
else
  clip_name = cell(1);
  clip_name{1} = "mask"; %% "026"; %% 
endif
disp(["clip_name{1} = ", clip_name{1}]);
distractor_mask_flag = ~target_mask_flag && mask_flag;
disp(["distractor_mask_flag = ", num2str(distractor_mask_flag)]);
if clip_flag
  version_ids = [1:16]; %%  
else
  version_ids = [1:16]; %% 
endif
disp(["version_ids = ", mat2str(version_ids)]);
pvp_frame_size = [1080 1920]; %%  [256, 256]; %%
disp(["frame_size = ", mat2str(pvp_frame_size)]);
pvp_edge_type = "canny"; 
pvp_clique_id = "3way"; %% ""; %% 
%% end definition of the most volitile parameters

%% version_str stores the training or testing run index 
num_versions = length(version_ids);
if num_versions > 0
  version_str = cell(num_versions,1);
  for i_version = 1 : num_versions
    version_str{i_version} = num2str(version_ids(i_version), "%3.3i");   
  endfor
else
  num_versions = 1;
  version_str = cell(num_versions,1);
  version_str{1}="";
endif

global PVP_VERBOSE_FLAG
PVP_VERBOSE_FLAG = 0;

global pvp_home_path
global pvp_workspace_path
global pvp_mlab_path
global pvp_clique_path
pvp_home_path = ...
    [filesep, "home", filesep, "gkenyon", filesep];
pvp_workspace_path = ...
    [pvp_home_path, "workspace-indigo", filesep];
pvp_mlab_path = ...
    [pvp_workspace_path, "PetaVision", filesep, "mlab", filesep];
pvp_clique_path = ...
    [pvp_workspace_path, "Clique2", filesep];


DATASET_ID = "Heli"; %% "amoeba"; %%"Tower"; %% "Tailwind"; %% 
dataset_id = tolower(DATASET_ID); %% 
flavor_id = tolower(FLAVOR_ID); %% 
pvp_repo_path = ...
    [filesep, "mnt", filesep, "data", filesep, "repo", filesep];
>>>>>>> refs/remotes/eclipse_auto/master
pvp_petavision_path = ...
    [pvp_repo_path, "neovision-programs-petavision", filesep]; %%, ...
pvp_dataset_path = ...
    [pvp_petavision_path, ...
     DATASET_ID, filesep]; %%, ...
mkdir(pvp_dataset_path);
pvp_flavor_path = ...
    [pvp_dataset_path, ...
     FLAVOR_ID, filesep]; %%, ...
mkdir(pvp_flavor_path);
pvp_program_path = ...
    pvp_flavor_path;
mkdir(pvp_program_path);

pvp_input_path2 = ...
    [pvp_clique_path, "input", filesep]; %%, ...
pvp_input_path3 = ...
    [pvp_input_path2, ...
     DATASET_ID, filesep]; %%, ...
mkdir(pvp_input_path3);
pvp_input_path = ...
    [pvp_input_path3, ...
     FLAVOR_ID, filesep];
mkdir(pvp_input_path);

pvp_num_ODD_kernels_str = "";
if pvp_num_ODD_kernels > 1
  pvp_num_ODD_kernels_str = num2str(pvp_num_ODD_kernels);
endif
pvp_bootstrap_str = ""; %% "_bootstrap0"; %%  
pvp_num_frames =  []; %% ceil(12294 / num_versions); %%625;

output_activity_path = ...
    [pvp_program_path, ...
     "activity", filesep];
mkdir(output_activity_path);

pvp_list_path = ...
    [pvp_program_path, ...
     "list", "_", pvp_edge_type, filesep];

%% path to generic image processing routines
util_dir = "~/workspace-indigo/PetaVision/mlab/util/";
addpath(util_dir);

for i_object = 1 : length(object_type)
  disp(object_type{i_object});
  
  pvp_params_template = ...
      [pvp_clique_path, ...
       "templates", filesep, ...
       DATASET_ID, filesep, ...
       FLAVOR_ID, filesep, ...
       object_type{i_object}, pvp_num_ODD_kernels_str, filesep, ...
       pvp_edge_type, pvp_clique_id, filesep, ...
       DATASET_ID, "_", FLAVOR_ID, "_", object_type{i_object}, pvp_num_ODD_kernels_str, "_", pvp_edge_type, pvp_clique_id, "_", ...
       "template.params"];
  
  output_object_path2 = ...
      [output_activity_path, ...
       object_type{i_object}, pvp_num_ODD_kernels_str, filesep];
  mkdir(output_object_path2);
  output_object_path = ...
      [output_object_path2, ...
       pvp_edge_type, pvp_clique_id, filesep];
  mkdir(output_object_path);

  input_object_path2 = ...
      [pvp_input_path, ...
       object_type{i_object}, pvp_num_ODD_kernels_str, filesep];
  mkdir(input_object_path2);
  input_object_path = ...
      [input_object_path2, ...
       pvp_edge_type, pvp_clique_id, filesep];
  mkdir(input_object_path);

  list_object_path = ...
      [pvp_list_path]; %%, ...
  %%object_type{i_object}, filesep];

  for i_clip = 1 : length(clip_name)
    disp(clip_name{i_clip});
    
    output_clip_path = ...
	[output_object_path, ...
	   clip_name{i_clip}, filesep];
    mkdir(output_clip_path);
    
    input_clip_path = ...
	[input_object_path, ...
	   clip_name{i_clip}, filesep];
    mkdir(input_clip_path);
    
    list_clip_path = ...
	[list_object_path, ...
	 clip_name{i_clip}, filesep];
    list_path = list_clip_path;

    for i_version = 1 : num_versions
      if num_versions > 1
	disp(version_str{i_version});
      
	output_version_path = ...
	    [output_clip_path, ...
	     version_str{i_version}, filesep];
	mkdir(output_version_path);
	output_path = output_version_path;
	
	input_version_path = ...
	    [input_clip_path, ...
	     version_str{i_version}, filesep];
	mkdir(input_version_path);
	input_path = input_version_path;
	
	pvp_fileOfFrames_path = ...
	    [list_path];
	pvp_fileOfFrames_file = ...
	    [clip_name{i_clip}, "_", version_str{i_version}, "_", "fileOfFilenames.txt"];
	pvp_fileOfFrames = ...
	    [pvp_fileOfFrames_path, pvp_fileOfFrames_file];
	pvp_num_frames = linecount(pvp_fileOfFrames);
	if pvp_num_frames == 0
	  error(["linecount = 0:", "pvp_fileOfFrames = ", pvp_fileOfFrames]);
	endif
	
	if target_mask_flag
	  pvp_fileOfMasks_file = ...
	      [clip_name{i_clip}, "_", version_str{i_version}, "_", "fileOfTargetMasknames.txt"];
	  pvp_fileOfMasks = ...
	      [pvp_fileOfFrames_path, pvp_fileOfMasks_file];
	  %%disp(["pvp_fileOfMasks: ", pvp_fileOfMasks]);
	  %%if ~exist("pvp_fileOfMasks", "file")
	elseif distractor_mask_flag
	  pvp_fileOfMasks_file = ...
	      [clip_name{i_clip}, "_", version_str{i_version}, "_", "fileOfDistractorMasknames.txt"];
	  pvp_fileOfMasks = ...
	      [pvp_fileOfFrames_path, pvp_fileOfMasks_file];	  
	else
	  pvp_fileOfMasks = [];
	endif
	params_filename = ...
	    [DATASET_ID, ...
	     "_", ...
	     FLAVOR_ID, ...
	     "_", ...
	     object_type{i_object}, ...
	     pvp_num_ODD_kernels_str, pvp_bootstrap_str, ...
	     "_", ...
	     pvp_edge_type, pvp_clique_id, ...
	     "_", ...
	     clip_name{i_clip}, ...
	     "_", ...
	     version_str{i_version}, ...
	     ".params"];


      else
	output_path = [output_clip_path];
	
	input_version_path = input_clip_path;
	input_path = input_version_path;
	
	pvp_fileOfFrames_path = ...
	    [list_path];
	pvp_fileOfFrames_file = ...
	    [clip_name{i_clip}, "_", "fileOfFilenames.txt"];
	pvp_fileOfFrames = ...
	    [pvp_fileOfFrames_path, pvp_fileOfFrames_file];
	pvp_num_frames = linecount(pvp_fileOfFrames);
	if pvp_num_frames == 0
	  error(["linecount = 0:", "pvp_fileOfFrames = ", pvp_fileOfFrames]);
	endif
	if target_mask_flag
	  pvp_fileOfMasks_file = ...
	      [clip_name{i_clip}, "_", "fileOfTargetMasknames.txt"];
	  pvp_fileOfMasks = ...
	      [pvp_fileOfFrames_path, pvp_fileOfMasks_file];
	  %%disp(["pvp_fileOfMasks: ", pvp_fileOfMasks]);
	  %%if ~exist("pvp_fileOfMasks", "file")
	elseif distractor_mask_flag
	  pvp_fileOfMasks_file = ...
	      [clip_name{i_clip}, "_", "fileOfDistractorMasknames.txt"];
	  pvp_fileOfMasks = ...
	      [pvp_fileOfFrames_path, pvp_fileOfMasks_file];	  
	else
	  pvp_fileOfMasks = [];
	endif
	params_filename = ...
	    [DATASET_ID, ...
	     "_", ...
	     FLAVOR_ID, ...
	     "_", ...
	     object_type{i_object}, ...
	     pvp_num_ODD_kernels_str, pvp_bootstrap_str, ...
	     "_", ...
	     pvp_edge_type, pvp_clique_id, ...
	     "_", ...
	     clip_name{i_clip}, ...
	     ".params"];
      endif  %% num_versions > 0

      [pvp_params_file] = ...
	  pvp_makeHeliParams(DATASET_ID, ...
			  FLAVOR_ID, ...
			  pvp_repo_path, ...
			  pvp_program_path, ...
			  input_path, ...
			  clip_name{i_clip}, ...
			  object_type{i_object}, ...
			  pvp_num_ODD_kernels, ...
			  pvp_bootstrap_str, ...
			  pvp_edge_type, ...
			  pvp_clique_id, ...
			  version_str{i_version}, ...
			  pvp_params_template, ...
			  pvp_frame_size, ...
			  pvp_num_frames, ...
			  list_path, ...
			  pvp_fileOfFrames, ...
			  pvp_fileOfMasks, ...
			  output_path, ...
			  params_filename);

      
    endfor
  endfor
endfor %% i_clip
