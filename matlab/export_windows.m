% export_windows.m
% Segment PPG into fixed-length windows for RNN training

clear; clc;

% Load processed PPG
load('data/ppg_processed.mat');  % ppg_all

fs = 125;                % sampling frequency (Hz)
window_sec = 5;          % window length in seconds
overlap_sec = 2.5;       % overlap length in seconds

window_len = window_sec * fs;      % window length in samples
overlap_len = overlap_sec * fs;    % overlap in samples

X_windows = {};
y_windows = {};   % optional labels (if available)

N = numel(ppg_all);

for i = 1:N
    ppg = ppg_all{i};
    len = length(ppg);
    
    start_idx = 1;
    while (start_idx + window_len - 1) <= len
        window = ppg(start_idx : start_idx + window_len - 1);
        
        % Store window
        X_windows{end+1,1} = window;
        
        % Optional label: replace with actual AF label per recording
        y_windows{end+1,1} = 0;  % 0 = Non-AF (replace with real label)
        
        % Move start index
        start_idx = start_idx + (window_len - overlap_len);
    end
end

% Convert to matrix if needed (samples x timesteps)
X_matrix = cell2mat(cellfun(@(x) x(:)', X_windows,'UniformOutput',false))';

% Save windows
save('data/ppg_windows.mat','X_windows','y_windows','X_matrix');
disp(['✅ Segmentation complete. Total windows: ', num2str(length(X_windows))]);
