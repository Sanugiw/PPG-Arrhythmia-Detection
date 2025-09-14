% preprocess_ppg.m
% Load raw PPG signal, handle NaNs, filter, normalize

load('data/mimic_perform_af_data.mat');   % loads struct array 'data'
N = numel(data);   % number of recordings
ppg_all = cell(N,1);

for i = 1:N
    % Use dataset-specific sampling frequency if available
    if isfield(data(i).ppg,'fs')
        fs = double(data(i).ppg.fs);  
    else
        fs = 125;  % fallback default
    end

    % Extract raw PPG values
    ppg_raw = double(data(i).ppg.v(:));

    % Handle NaN or Inf values by linear interpolation
    if any(~isfinite(ppg_raw))
        t = 1:numel(ppg_raw);
        ppg_raw(~isfinite(ppg_raw)) = interp1(t(isfinite(ppg_raw)), ppg_raw(isfinite(ppg_raw)), t(~isfinite(ppg_raw)), 'linear', 'extrap');
    end

    % Band-pass filter (0.5–8 Hz for PPG)
    [b,a] = butter(3, [0.5 8]/(fs/2), 'bandpass');
    ppg_filt = filtfilt(b, a, ppg_raw);

    % Normalize (z-score)
    ppg_norm = (ppg_filt - mean(ppg_filt)) / std(ppg_filt);

    % Store in cell array
    ppg_all{i} = ppg_norm;
end

% Save all processed signals
save('data/ppg_processed.mat','ppg_all');
disp('Preprocessing complete. All signals saved in data/ppg_processed.mat');



