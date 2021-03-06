function [pvp_params_file] = ...
      pvp_makeAmoebaParams(DATASET_ID, ...
		      FLAVOR_ID, ...
		      pvp_repo_path, ...
		      pvp_program_path, ...
		      pvp_input_path, ...
		      pvp_clip_name, ...
		      pvp_object_type, ...
		      pvp_num_ODD_kernels, ...
		      pvp_bootstrap_str, ...
		      pvp_edge_type, ...
		      pvp_version_str, ...
		      pvp_params_template, ...
		      pvp_frame_size, ...
		      pvp_num_frames, ...
		      pvp_list_path, ...
		      pvp_fileOfFrames, ...
		      pvp_fileOfMasks, ...
		      pvp_output_path, ...
		      pvp_params_filename)
  
  %%keyboard;
  global PVP_VERBOSE_FLAG
  if ~exist("PVP_VERBOSE_FLAG") || isempty(PVP_VERBOSE_FLAG)
    PVP_VERBOSE_FLAG = 0;
  endif
  global pvp_home_path
  global pvp_workspace_path
  global pvp_mlab_path
  global pvp_clique_path
  if isempty(pvp_home_path)
    pvp_home_path = ...
	[filesep, "home", filesep, "garkenyon", filesep];
    %%[filesep, "Users", filesep, "gkenyon", filesep];
  endif
  if isempty(pvp_workspace_path)
    pvp_workspace_path = ...
	[pvp_home_path, "workspace-indigo", filesep];
  endif
  if isempty(pvp_mlab_path)
    pvp_mlab_path = ...
	[pvp_home_path, "workspace-indigo", filesep, "PetaVision", filesep, "mlab", filesep];
  endif
  if isempty(pvp_clique_path)
    pvp_clique_path = ...
	[pvp_workspace_path, "Clique2", filesep];
  endif

  more off;
  begin_time = time();

  num_argin = 0;
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("DATASET_ID") || isempty(DATASET_ID)
    DATASET_ID = "amoeba"; %%"Heli"; %% "Tower"; %% "Tailwind"; %% 
  endif
  dataset_id = tolower(DATASET_ID); %% 
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("FLAVOR_ID") || isempty(FLAVOR_ID)
    FLAVOR_ID = "33x33"; %% "3way"; %%"Training"; %% "Challenge"; %% "Formative"; %%  
  endif
  flavor_id = tolower(FLAVOR_ID); %% 
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_repo_path") || isempty(pvp_repo_path)
    pvp_repo_path = ...
	[filesep, "mnt", filesep, "data", filesep];
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_program_path") || isempty(pvp_program_path)
    pvp_program_path = ...
	[pvp_repo_path, "PetaVision", filesep, ...
	 DATASET_ID, filesep, ...
	 FLAVOR_ID, filesep];
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_input_path") || isempty(pvp_input_path)
    pvp_input_path = ...
	[pvp_clique_path, "input", filesep, ...
	 DATASET_ID, filesep, ...
	 FLAVOR_ID, filesep];
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_clip_name") || isempty(pvp_clip_name)
    pvp_clip_name = "t"; %% "045"; %%
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_object_type") || isempty(pvp_object_type)
    pvp_object_type =  "4FC"; %% "Car"; %%
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_num_ODD_kernels") || isempty(pvp_num_ODD_kernels)
    pvp_num_ODD_kernels =  1; %% 3; %%
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_bootstrap_str") %% string can be empty
    pvp_bootstrap_str =  ""; %% "_bootstrap"; %%
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_edge_type") || isempty(pvp_edge_type)
    pvp_edge_type =  ""; %% "canny"; %%
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_version_str") %% string can be empty
    pvp_version_str =  "1"; %%
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_params_template") || isempty(pvp_params_template)
    pvp_params_template = ...
	[pvp_clique_path, ...
	 "templates", filesep, ...
	 DATASET_ID, "_", FLAVOR_ID, "_", ...
	 pvp_object_type, num2str(pvp_num_ODD_kernels), pvp_bootstrap_str, "_", pvp_edge_type, pvp_version_str,  "_", ...
	 "template.params"];
  endif
  clip_log_struct = struct;
  clip_log_struct.tot_clips = 625; %%450;
  clip_log_struct.ave_cropped_size = [256, 256]; %%[1080 1920];
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_frame_size") || isempty(pvp_frame_size)
    pvp_frame_size =  clip_log_struct.ave_cropped_size; %%
    disp(["frame_size = ", num2str(pvp_frame_size)]);
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_num_frames") || isempty(pvp_num_frames)
    pvp_num_frames =  clip_log_struct.tot_clips; %%
    disp(["num_frames = ", num2str(pvp_num_frames)]);
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_list_path") || isempty(pvp_list_path)
    pvp_list_path = ...
	[pvp_program_path, 
	 "list", pvp_edge_type, filesep];
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_fileOfFrames") || isempty(pvp_fileOfFrames)
    pvp_fileOfFrames_path = ...
	[list_path, pvp_clip_name, filesep];
    pvp_fileOfFrames_file = ...
	[pvp_clip_name, "_", "001", "_", "fileOfFilenames.txt"];
	%%[DATASET_ID, "_", FLAVOR_ID, "_", pvp_clip_name, "_", pvp_edge_type, "_", "frames"];
    pvp_fileOfFrames = ...
	[pvp_fileOfFrames_path, pvp_fileOfFrames_file];
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_fileOfMasks") || isempty(pvp_fileOfMasks)
    pvp_fileOfMasks_path = ...
	[list_path, pvp_clip_name, filesep];
    pvp_fileOfMasks_file = ...
	[pvp_clip_name, "_", "1", "_", "fileOfFilenames.txt"];
	%%[DATASET_ID, "_", FLAVOR_ID, "_", pvp_clip_name, "_", pvp_edge_type, "_", "frames"];
    pvp_fileOfMasks = ...
	[pvp_fileOfMasks_path, pvp_fileOfMasks_file];
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_output_path") || isempty(pvp_output_path)
    output_activity_path = ...
	[pvp_program_path, ...
	 "activity", filesep];
    mkdir(output_activity_path);
    output_object_path = ...
	[output_activity_path, ...
	 pvp_object_type, filesep];
    mkdir(output_object_path);
    output_clip_path = ...
	[output_object_path, ...
	 pvp_clip_name, filesep];
    mkdir(output_clip_path);
    output_version_path = ...
	[output_clip_path, ...
	 pvp_version_str, filesep];
    mkdir(output_version_path);
    pvp_output_path = output_version_path;
  endif
  num_argin = num_argin + 1;
  if nargin < num_argin || ~exist("pvp_params_filename") || isempty(pvp_params_filename)
    pvp_params_filename = ...
	[DATASET_ID, ...
	 "_", ...
	 FLAVOR_ID, ...
	 "_", ...
	 pvp_object_type, ...
	 num2str(pvp_num_ODD_kernels), pvp_bootstrap_str, ...
	   "_", ...
	 pvp_clip_name, ...
	 "_", ...
	 pvp_edge_type, ...
	 pvp_version_str, ...
	 ".params"];
  endif
  
  pvp_params_file = [pvp_input_path, pvp_params_filename];
  pvp_params_fid = ...
      fopen(pvp_params_file, "w");
  if pvp_params_fid < 0
    disp(["fopen failed: ", pvp_params_file]);
    return;
  end
  
  pvp_template_fid = ...
      fopen(pvp_params_template, "r");
  if pvp_template_fid < 0
    disp(["fopen failed: ", pvp_params_template]);
    return;
 end

  pvp_params_token_left = "$$$_";
  pvp_params_token_right = "_$$$";
  pvp_params_hash = ...
      {"numSteps", "numSteps", num2str(pvp_num_frames + pvp_num_ODD_kernels + 4); ...
       "outputPath", "outputPath", ["""", pvp_output_path, """"]; ...
       "imageListPath", "imageListPath", ["""", pvp_fileOfFrames, """"]; ...
       "maskListPath", "imageListPath", ["""", pvp_fileOfMasks, """"]; ...
       "endStim", "endStim", num2str(pvp_num_frames); ...
       "VgainL1Clique", "Vgain", num2str(0.5); ...
       "VgainL2Clique", "Vgain", num2str(0.5); ...
       "VgainL3Clique", "Vgain", num2str(0.5); ...
       "VgainL4Clique", "Vgain", num2str(0.5); ...
       };
%%  pvp_params_hash = ...
%%      {"nx", "nx",num2str(pvp_frame_size(2)); ...
%%       "ny", "ny", num2str(pvp_frame_size(1)); ...
%%       "numSteps", "numSteps", num2str(pvp_num_frames + pvp_num_ODD_kernels + 4); ...
%%       "outputPath", "outputPath", ["""", pvp_activity_path, """"]; ...
%%       "imageListPath", "imageListPath", ["""", pvp_fileOfFrames, """"]; ...
%%       "burstDuration", "burstDuration", num2str(pvp_num_frames); ...
%%       "endStim", "endStim", num2str(pvp_num_frames); ...
%%       "VgainL1Clique", "Vgain", num2str(0.03125); ...
%%       "VgainL2Clique", "Vgain", num2str(0.0625); ...
%%       "VgainL3Clique", "Vgain", num2str(0.0625); ...
%%       "VgainL4Clique", "Vgain", num2str(0.0625); ...
%%       };
  pvp_num_params = size(pvp_params_hash, 1);
       
  %%keyboard;
  while(~feof(pvp_template_fid))
    pvp_template_str = fgets(pvp_template_fid);
    pvp_params_str = pvp_template_str;
    for pvp_params_ndx = 1 : pvp_num_params
      pvp_str_ndx = ...
	  strfind(pvp_template_str, ...
		  [pvp_params_token_left, ...
		   pvp_params_hash{pvp_params_ndx, 1}, ...
		   pvp_params_token_right]);
      if ~isempty(pvp_str_ndx)
	pvp_hash_len = ...
	    length(pvp_params_hash{pvp_params_ndx, 1}) + ...
	    length(pvp_params_token_left) + ...
	    length(pvp_params_token_right);
	pvp_template_len = ...
	    length(pvp_template_str);
	pvp_prefix = pvp_template_str(1:pvp_str_ndx-1);
	pvp_suffix = pvp_template_str(pvp_str_ndx+pvp_hash_len:pvp_template_len-1);
	pvp_params_str = ...
	    [pvp_prefix, ...
	     pvp_params_hash{pvp_params_ndx, 2}, ...
	     " = ", ...
	     num2str(pvp_params_hash{pvp_params_ndx, 3}), ...
	     pvp_suffix, ";", "\n"];
	break;
      endif
    endfor  %% pvp_params_ndx
    fputs(pvp_params_fid, pvp_params_str);
    %%keyboard;
  endwhile
  fclose(pvp_params_fid);
  fclose(pvp_template_fid);

endfunction %% pvp_makeParams
