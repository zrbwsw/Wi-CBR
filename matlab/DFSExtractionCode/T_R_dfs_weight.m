% 根据频移分布，计算时间权重，接收器权重

% File paths and parameters
filepath = 'D:\Widar3.0\CSI\20181130_user5_10_11\user10\'; % Raw CSI path
sapath = 'D:\Widar3.0\T_R_DFS_weight\'; % Save path for DFS images
uname = 'user10'; % User name
suname = '1'; % Save file name prefix
rx_cnt = 6; % Number of receivers
rx_acnt = 3; % Number of antennas per receiver
method = 'stft'; % Use STFT for time-frequency analysis

% Ensure output directory exists
if ~exist(sapath, 'dir')
    mkdir(sapath);
end

% Nested loops over CSI file indices
for mn = 3
    for ln = 1
        for on = 3
            for rn = 1
                spfx_ges = [filepath, uname, '-', num2str(mn), '-', num2str(ln), '-', num2str(on), '-', num2str(rn)];
                try
                    [doppler_spectrum, freq_bin] = get_doppler_spectrum(spfx_ges, rx_cnt, rx_acnt, method);
                catch err
                    disp(['Error processing ', spfx_ges, ': ', err.message]);
                    continue;
                end
                
                % 初始化频移总量二维数组[6,T]
                time_steps = size(doppler_spectrum, 3);
                freq_weight = zeros(rx_cnt, time_steps); % 存储每个接收器每个时间点的频移总量
                
                % 计算每个接收器的频移总量
                for rsn = 1:rx_cnt
                    dfs = squeeze(doppler_spectrum(rsn, :, :));
                    
                    % FFT频移对齐
                    dfs_shifted = fftshift(dfs, 1);
                    freq_bin_shifted = fftshift(freq_bin);
                    
                    % 计算每个时间戳的频移总量（加权和）
                    for t = 1:time_steps
                        % 提取当前时间点的频移分布
                        freq_slice = dfs_shifted(:, t);
                        
                        % 计算加权和：频率值 × 对应幅值（取绝对值加权）
                        weighted_sum = sum(abs(freq_bin_shifted) .* freq_slice');
                        freq_weight(rsn, t) = weighted_sum;
                    end
                end                
            
                % 1. 时间维度插值：将T步压缩到224步
                freq_weight = imresize(freq_weight, [rx_cnt, 224], 'bilinear'); 
                %　按照每一个接收器，6个值，归一化

                % 2. 接收器维度插值：将6个接收器扩展到224行
                freq_weight_resized = imresize(freq_weight, [224, 224], 'bilinear');

                % 保存结果
                save_name = [suname, '-', num2str(mn), '-', num2str(ln), '-', num2str(on), '-', num2str(rn), '.mat'];
                save(fullfile(sapath, save_name), 'freq_weight_resized');
                disp(['Successfully saved: ', save_name]);
            end
        end
    end
end