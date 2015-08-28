% [t,reconerror,costfunc,totalenergy,activationpercent,inputenergy,timescale] = readSoftThreshold()
function [t,reconerror,costfunc,totalenergy,activationpercent, inputenergy, timescale] = ...
   readSoftThreshold()

[t, reconerror] = ...
    readenergydata('../output-SoftThreshold/recon_error_l2norm.txt',...
                   't = %f b = 0 numNeurons = %d L2-norm squared = %f',...
                   [1 3]);

[t, costfunc] = ...
    readenergydata('../output-SoftThreshold/cost_function.txt',...
                   't = %f b = 0 numNeurons = %d L1-norm = %f',...
                   [1 3]);

[t, totalenergy] = ...
    readenergydata('../output-SoftThreshold/total_energy.txt',...
                   '"Total_Energy_Probe",%f,0,%f\n',...
                   [1 2]);

[~, numneurons] = ...
    readenergydata('../output-SoftThreshold/activation_percentage.txt',...
                   't = %f b = 0 numNeurons = %d L0-norm = %f',...
                   [1 2]);

[t, activation] = ...
    readenergydata('../output-SoftThreshold/activation_percentage.txt',...
                   't = %f b = 0 numNeurons = %d L0-norm = %f',...
                   [1 3]);
activationpercent = activation./numneurons;

[t, inputenergy] = ...
    readenergydata('../output-SoftThreshold/input_energy.txt',...
                   '"Input_Energy",%f,0,%f',...
                   [1 2]);

[t, timescale] = ...
    readenergydata('../output-SoftThreshold/timescale.txt',...
                   'Scaled_Energy,%f,0,%f',...
                   [1 2]);
