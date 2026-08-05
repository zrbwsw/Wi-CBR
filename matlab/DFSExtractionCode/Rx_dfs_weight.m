% File paths and parameters
filepath = ''; 
sapath = ''; % Save path for DFS images
uname = 'user5'; % User name
suname = '0'; % Save file name prefix
rx_cnt = 6; % Number of receivers
rx_acnt = 3; % Number of antennas per receiver
method = 'stft'; % Use STFT for time-frequency analysis

% Ensure output directory exists
if ~exist(sapath, 'dir')
    mkdir(sapath);
end

% Nested loops over CSI file indices
for mn = 1:9
    for ln = 1:5
        for on = 1:5
            for rn = 1:5
                spfx_ges = [filepath, uname, '-', num2str(mn), '-', num2str(ln), '-', num2str(on), '-', num2str(rn)];
                try
                    [doppler_spectrum, freq_bin] = get_doppler_spectrum(spfx_ges, rx_cnt, rx_acnt, method);
                catch err
                    disp(['Error processing ', spfx_ges, ': ', err.message]);
                    continue;
                end
                
                time_steps = size(doppler_spectrum, 3);
                freq_weight = zeros(rx_cnt, time_steps); 
                
                for rsn = 1:rx_cnt
                    dfs = squeeze(doppler_spectrum(rsn, :, :));
                    
                    dfs_shifted = fftshift(dfs, 1);
                    freq_bin_shifted = fftshift(freq_bin);
                    
                    for t = 1:time_steps
                        freq_slice = dfs_shifted(:, t);
                        
                        weighted_sum = sum(abs(freq_bin_shifted) .* freq_slice');
                        freq_weight(rsn, t) = weighted_sum;
                    end
                end                
            
                freq_weight = imresize(freq_weight, [rx_cnt, 224], 'bilinear'); 
                
                save_name = [suname, '-', num2str(mn), '-', num2str(ln), '-', num2str(on), '-', num2str(rn), '.mat'];
                save(fullfile(sapath, save_name), 'freq_weight');
                disp(['Successfully saved: ', save_name]);
            end
        end
    end
end
