filepath = 'E:\XRF55\Scene_4\';
sapath = 'E:\XRF55\QFM\DFS_images\';

rx_cnt = 3;        % Total number of receivers
rx_acnt = 3;       % Number of active receivers
method = 'stft';   % Spectral analysis method

% Create output directory
if ~exist(sapath,'dir')
    mkdir(sapath);
end

receivers = {'lb', 'lf', 'rb'}; % Receiver prefixes
% ========== User configuration corrected version ==========
user_config = struct(...
    'input_ids', {{'03', '04', '13'}}, ... % Note double braces wrapping
    'output_base', 37   ...
);
% ========== Loop structure optimization ==========
for user_idx = 1:numel(user_config.input_ids) % Changed to use numel to get element count
    % Input path processing
    raw_user_id = user_config.input_ids{user_idx}; % Use {} to access cell elements
    
    % Auto zero-padding processing
    if length(raw_user_id) == 1
        user_folder = ['0', raw_user_id];
    else
        user_folder = raw_user_id;
    end
             
    % Output ID mapping (Scene_2 starts from 31)
    output_user_id = user_config.output_base + (user_idx - 1);
    
    % Action loop
    for mn = 1:55
        mn_str = sprintf('%02d', mn);
        
        % Repetition loop
        for rn = 1:20
            rn_str = sprintf('%02d', rn);
            
            % Receiver loop
            for rsn = 1:3
                % Build complete path
                receiver = receivers{rsn};
                user_dir = fullfile(filepath, receiver, user_folder);
                filename = fullfile(user_dir, ...
                    sprintf('%s_%s_%s.dat', user_folder, mn_str, rn_str));
                
                % File existence check
                if ~exist(filename, 'file')
                    fprintf('[Skip] File does not exist: %s\n', filename);
                    continue;
                end
                try
                    % Call modified processing function
                    [doppler_spectrum, freq_bin] = xrf_get_doppler_spectrum(filename, rx_cnt, rx_acnt, method);
                catch err
                    disp(['Error processing ', filename, ': ', err.message]);
                    continue;
                end
                % ==== Spectrum processing ====
                dfs = squeeze(doppler_spectrum(1, :, :)); % Modification point 1: Use fixed dimension index
                dfs_shifted = fftshift(dfs, 1);
                
                % ==== Visualization and saving ====
                fmi = figure('visible', 'off');
                imagesc(dfs_shifted);
                axis off;
                set(gca, 'Position', [0 0 1 1]);
                
                % Generate output filename (maintain original format)
                sname = [num2str(output_user_id), '-', num2str(mn), '-', num2str(rn), '-', num2str(rsn)];
                saveas(fmi, fullfile(sapath, [sname, '.jpg']));
                disp(['Save successful: ', sname]);
                close(fmi);    % Close figure window
                delete(fmi);   % Delete figure object
                clear fmi;     % Clear variable reference
            end
        end
    end
end