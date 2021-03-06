%
% Generates STDP params files
%

clear all; close all; more off; clc;
system("clear");
warning off;
format short g;

global MOVIE_FLAG = 0;
global PRINT_FLAG;
global SWEEP_POS = 0;

addpath('/Users/rcosta/Documents/workspace/HyPerSTDP/analysis/');
setenv("GNUTERM", "x11");

fullOrient_DATASET = "orient_36r";
DATASET = "OlshausenField_whitened12x12_tinyAll1000"; %%orient_36r orient_simple  OlshausenField_raw32x32_tiny OlshausenField_raw12x12_tinyAll
TEMPLATE = "STDP3generalNS";
on_v1_file = "w4_post.pvp";
off_v1_file = "w5_post.pvp";

numsteps = 5000;

%Masquelier params
%pr = 17*0.01/34*0.0085;


%% INDEXES
DISPLAYPERIODi=7;
STRENGTH_IMAGE2RETINAi = 8;

wMaxInitSTDPi = 9;
wMinInitSTDPi = 10;
tauLTPi = 11;
tauLTDi = 12;
ampLTPi = 13;
ampLTDi = 14;
wMini = 15;
wMaxi = 16;
synscalingi = 17;
synscalingvi = 18;
tauYi = 19;

RUN_FLAG = 1;

PARAMSWEEP_FLAG = 0;
%PARAM_SWEEP = [STRENGTH_IMAGE2RETINAi synscalingvi]
PARAM_SWEEP = [tauLTPi tauLTDi ampLTPi ampLTDi];
MEASURES_FLAG = 0;
MEASURES_PLOT_FLAG = 0;
MEASURES_OSI_FLAG = 1;
MEASURES_GM_FLAG = 0;

global ROTATE_FLAG; ROTATE_FLAG = 0;

v1_cells = 6*6;
img_size = 12;
ign_w = 1;

global params;
params{1} = "false";  %checkpointRead
params{2} = "true";  %checkpointWrite
params{3} = DATASET;  %checkpointReadDir
params{4} = 0;  %checkpointReadDirIndex
params{5} = 1000;  %checkpointWriteStepInterval
params{6} = "true"; %plasticityFlag
params{DISPLAYPERIODi} = 20; %displayPeriod (image display period)
params{STRENGTH_IMAGE2RETINAi} = 15;



%STDP params
%Natural images params
params{wMaxInitSTDPi} = 0.5;
params{wMinInitSTDPi} = 0.005;
params{tauLTPi} = 16.8;
params{tauLTDi} = 33.7;
params{ampLTPi} = 0.05001;
params{ampLTDi} = 0.0065;
params{wMini} = 0.001;
params{wMaxi} = 7;
params{synscalingi} = 1;
params{synscalingvi} = 10;
params{tauYi} = 114;

LOAD_FILE = 0;
if(LOAD_FILE)
    load('rg_p_NS');
    for x=1:size(rg_p,2) %Set params
    params{PARAM_SWEEP(x)} = rg_p(135,x);
    end
end


if(PARAMSWEEP_FLAG)
%Range over which 
rg = {1:10:30; ... 
    1:10:30; ...
    0.09:0.01:0.2; ...
    0.0055:0.001:0.01};  %tauLTP, tauLTD, ampLTP, ampLTD, synscalingvi
%rg = {1; ... 
    %      37; ...
    %      0.115; ...
    %      0.0085};  %tauLTP, tauLTD, ampLTP, ampLTD
%Generates all combinations
rg_p = allcomb(rg{:});

%rg_p = rg_p(1:3,:);

disp("---------------------------");
disp("------Parameter Sweep------");
disp("---------------------------");

OSIm = zeros(size(rg_p,1),2);
pr = zeros(size(rg_p,1),1);
%keyboard
else
rg = params{PARAM_SWEEP(1)};
rg_p = params{PARAM_SWEEP(1)};
OSIm = zeros(1,2);
pr = zeros(1,1);
end

tic
for ps=1:size(rg_p,1)

if(PARAMSWEEP_FLAG)
disp("--------------------------------");
disp(["--- Paramsweep: " num2str(ps) " out of " num2str(size(rg_p,1)) " ---"]);
disp("--------------------------------");
end

for x=1:size(rg_p,2) %Set params
params{PARAM_SWEEP(x)} = rg_p(ps,x);
end
rg_p(ps,:)
%OSIm(1:ps,:)
%keyboard

[pvp_params_file pvp_project_path pvp_output_path] = pvp_makeSTDP3Params(DATASET, [], TEMPLATE, [], numsteps);

if(RUN_FLAG)
if(PARAMSWEEP_FLAG)
system([pvp_project_path "Debug/HyPerSTDP -p " pvp_params_file " &> batchOutput.txt"]);
else
system([pvp_project_path "Debug/HyPerSTDP -p " pvp_params_file]);
end
end


addpath([pvp_project_path, "mlab"]);

if(PARAMSWEEP_FLAG==0)
PRINT_FLAG = 1;
SWEEP_POS = 0;
filename = [pvp_output_path, filesep, on_v1_file];
[data hdr wm]=readpvpfile(filename, [pvp_output_path, filesep], on_v1_file, 1);
else
PRINT_FLAG = 1;
SWEEP_POS = ps;
filename = [pvp_output_path, filesep, on_v1_file];
[data hdr wm]=readpvpfile(filename, [pvp_output_path, filesep], on_v1_file, 1);
end

PRINT_FLAG = 0;


%Plasticity ratio
pr(ps) = (params{tauLTDi}*params{ampLTDi})/params{tauLTPi}*params{ampLTPi};




if(MEASURES_FLAG)

if(MEASURES_OSI_FLAG)


if(PARAMSWEEP_FLAG==0)
disp("---------------------------");
disp("-------Tunning curves------");
disp("---------------------------");
end

%1. For each checkpoint/weight matrix

params{1} = "true";  %checkpointRead
params{2} = "false";  %checkpointWrite
params{3} = DATASET;  %checkpointReadDir
%params{5} = 100;  %checkpointWriteStepInterval
params{6} = "false";  %plasticityFlag

global PVP_VERBOSE_FLAG;
PVP_VERBOSE_FLAG = 0;

%Read Orient dataset
fid = fopen([pvp_project_path, "input", filesep, fullOrient_DATASET, '.txt' ], 'r');
datasetl = {};

c=1;
while(~feof(fid))
datasetl{c} = fgets(fid);
c=c+1;
end

if(PARAMSWEEP_FLAG==0)
%Get and plot weights
filename = [pvp_output_path, filesep, on_v1_file];
[data hdr wm]=readpvpfile(filename, [pvp_output_path, filesep], on_v1_file, 1);
end

hist_per_orient = zeros(numsteps/params{5}+1, v1_cells, length(datasetl));

for i=0:params{5}:numsteps
params{4} = i;  %checkpointReadDirIndex

%Generates new params file
[pvp_params_file pvp_project_path pvp_output_path] = pvp_makeSTDP3Params(fullOrient_DATASET, [], TEMPLATE, [], i + length(datasetl)*params{DISPLAYPERIODi});
length(datasetl)*params{DISPLAYPERIODi}+i

%pause
if(RUN_FLAG)
if(PARAMSWEEP_FLAG)
system([pvp_project_path "Debug/HyPerSTDP -p " pvp_params_file  " &> batchOutput.txt"]); %Runs new params file
else
system([pvp_project_path "Debug/HyPerSTDP -p " pvp_params_file]); %Runs new params file
end

%Reads V1 activity file (TODO: assumes that writing step for V1 is 1ms)
[data hdr] = readpvpfile([pvp_output_path, filesep, "S1.pvp"], [pvp_output_path, filesep], "S1.pvp");

for p=0:(length(datasetl)-1)
for f=1:params{DISPLAYPERIODi}
for v=1:v1_cells
hist_per_orient(i/params{5}+1,v,p+1) = hist_per_orient(i/params{5}+1,v,p+1) + data{p*params{DISPLAYPERIODi}+f}(v);
end
end
end
end
end
avg_r_per_orient = (hist_per_orient*1000)./params{7};

if(MEASURES_PLOT_FLAG)
%Prepare Plot
pict_size = 20;
orient_plot = zeros(size(data{1},1)*size(data{1},2)*pict_size, (numsteps/params{5})*pict_size);

%Build image
for i=0:size(data{1},1)*size(data{1},2)-1
for j=0:numsteps/params{5}-1
for z=1:size(avg_r_per_orient,3)
im = imread(strtrim(datasetl{z}));
orient_plot(i*pict_size+1:i*pict_size+pict_size, j*pict_size+1:j*pict_size+pict_size) += imresize(im, [pict_size pict_size]).*avg_r_per_orient(j+1, i+1, z);
end
end
end
figure
imagesc([0  (1/size(data{1},1)*size(data{1},2))*(numsteps/params{5})], [1 size(data{1},1)*size(data{1},2)], orient_plot);
axis image;
colorbar;
%set(gca,'XTick',[],'YTick',[])
end

% V1on = "w4_post.pvp";
% [data hdr wm] = readpvpfile([pvp_output_path, filesep, V1on], pvp_output_path, V1on, 1);

%V1off = "w5_post.pvp";
%[data hdr wm] = readpvpfile([pvp_output_path, V1off], pvp_output_path, V1off, 1);







if(PARAMSWEEP_FLAG==0)
disp("------------------------------------------");
disp("-------Orientation Selectivity Index------");
disp("------------------------------------------");
end


%Get Orientation Selectivity Index (OSI)
osi = zeros(v1_cells,1);
for x=1:length(osi) %Cell x
f=fft(sum(hist_per_orient(:,x,:),1));
%abs(f(2))
osi(x) = (abs(f(2))/(abs(f(2))+mean(mean(avg_r_per_orient(:,x,:),1))))*100; %A 2nd harmonic/(A 2nd harmonic + delta_firingrate);
end
osi(isnan(osi))=0;

if(MEASURES_PLOT_FLAG)
figure
hist(osi,30)
xlabel('Orientation Selectivity Index');
ylabel('Count');
box off;

plotHistOSI(hist_per_orient, 1, osi(1),36)
end

OSIm(ps,:) = [mean(osi) std(osi)];

%TODO: Plot OSI over time


end





if(MEASURES_GM_FLAG)


disp("------------------------------------------");
disp("------------Generative Measure------------");
disp("------------------------------------------");


%Read Orient dataset
fid = fopen([pvp_project_path, "input", filesep, DATASET, '.txt' ], 'r');
datasetl = {};

c=1;
while(~feof(fid))
datasetl{c} = fgets(fid);
c=c+1;
end

post = 1;
ign_w = 4;
hist_per_img = zeros(numsteps/params{5}+1, v1_cells, length(datasetl));
diff = zeros(numsteps/params{5}+1, length(datasetl));

%Generative measure
%1. For each image matrix
for i=numsteps-params{5}:params{5}:numsteps
%    for i=0:params{5}:numsteps
params{4} = i;  %checkpointReadDirIndex

%Generates new params file
[pvp_params_file pvp_project_path pvp_output_path] = pvp_makeSTDP3Params(DATASET, [], TEMPLATE, [], length(datasetl)*params{DISPLAYPERIODi}+i);

length(datasetl)*params{DISPLAYPERIODi}+i
%pause

if(RUN_FLAG)
system([pvp_project_path "Debug/HyPerSTDP -p " pvp_params_file]); %Runs new params file

%2. Get activity and weight matrix
%Reads V1 activity file (TODO: assumes that writing step for V1 is 1ms)
[data hdr] = readpvpfile([pvp_output_path, filesep, "S1.pvp"], [pvp_output_path, filesep], "S1.pvp");

%Reads the weights Retina_ON > V1 for the time being
[d hdr wm] = readpvpfile([pvp_output_path, filesep, on_v1_file], [pvp_output_path, filesep],on_v1_file, post);
figure
imshow(wm);
for p=0:(length(datasetl)-1)
for f=1:params{DISPLAYPERIODi}
for v=1:v1_cells
hist_per_img(i/params{5}+1,v,p+1) = hist_per_img(i/params{5}+1,v,p+1) + data{p*params{DISPLAYPERIODi}+f}(v);
end
end
end

%keyboard

%3. Reconstruct the original image
for p=1:(length(datasetl)) %Loop over images
img_recons = zeros(img_size);
for v=1:v1_cells %Loop over cells
mean_act = mean(hist_per_img(i/params{5}+1,v,p));
if(mean_act>0) %If cell reconstruct
[r c] = ind2sub([sqrt(v1_cells) sqrt(v1_cells)], v);
w=wm((r-1)*hdr.nxp+1:r*hdr.nxp, (c-1)*hdr.nyp+1:c*hdr.nyp);
w=w((hdr.nxp-(ign_w*(r-1)-1))-img_size:(hdr.nxp-(ign_w*(r-1))), (hdr.nxp-(ign_w*(c-1)-1))-img_size:(hdr.nxp-(ign_w*(c-1)))); %Get actual weights that are changed
img_recons = img_recons .+ (mean_act .* w);
[r c mean_act]
end
end

if(sum(sum(img_recons))>0)
img_recons = img_recons./max(max(img_recons));
img_orig = imread(strtrim(datasetl{p}));
if(ROTATE_FLAG==0)
img_orig = flipud(rot90(img_orig));
%img_orig = rot90(img_orig);
end
diff(i/params{5}+1,p) = mean(mean(abs(img_orig-img_recons)));
%if(i==numsteps)%Only plot the last ones
figure
subplot(1,2,1);
imshow(img_orig);
title('Original');
subplot(1,2,2);
imshow(img_recons);
title(['Reconstruction  diff=' num2str(diff(i/params{5}+1,p))]);
keyboard
%end

end
end    

%4. Generative measure (use KL divergence): mean(D(I_G,I_O))

end
end

end

end

end

if(PARAMSWEEP_FLAG)
%[rg_p pr OSIm]
%Print best
%rg_p(OSIm==max(OSIm(:,1)),:)
else
%[pr OSIm]
end     

toc
