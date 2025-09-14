% detect_beats.m
% Detect systolic peaks in PPG and compute inter-beat intervals (IBIs)

clear; clc;

% Load processed PPG signals
load('data/ppg_processed.mat');  % contains ppg_all

fs = 125;  % sampling frequency (adjust if you have subject-specific fs)
N = numel(ppg_all);

% Initialize cell arrays to store results
peak_locs_all = cell(N,1);
ibi_all = cell(N,1);

for i = 1:N
    ppg = ppg_all{i};
    
    % Detect peaks (systolic peaks)
    % 'MinPeakHeight' and 'MinPeakDistance' can be tuned
    [pks, locs] = findpeaks(ppg, ...
                            'MinPeakHeight', 0.5*max(ppg), ...
                            'MinPeakDistance', round(0.4*fs));
    
    % Store peaks
    peak_locs_all{i} = locs;
    
    % Compute inter-beat intervals (IBI) in seconds
    ibi = diff(locs) / fs;
    ibi_all{i} = ibi;
    
    % Optional: display number of beats
    fprintf('Recording %d: Detected %d beats\n', i, length(locs));
end

% Save peaks and IBIs
save('data/ppg_beats.mat','peak_locs_all','ibi_all');
disp('Beat detection complete. Results saved in data/ppg_beats.mat');
