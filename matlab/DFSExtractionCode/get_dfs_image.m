% DFS Visualization: Generate Doppler Frequency Shift images for each receiver

% File paths and parameters
filepath = 'D:\Widar3.0\CSI\20181204\20181204\user1\'; % Raw CSI path
sapath = 'D:\Widar3.0\DFS_images\'; % Save path for DFS images
uname = 'user1'; % User name
suname = '16'; % Save file name prefix
rx_cnt = 6; % Number of receivers
rx_acnt = 3; % Number of antennas per receiver
method = 'stft'; % Use STFT for time-frequency analysis

% Ensure output directory exists
if ~exist(sapath, 'dir')
    mkdir(sapath);
end

% Nested loops over CSI file indices
for mn = 1:6
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
                % Generate and save DFS image for each receiver
                for rsn = 1:rx_cnt
                    dfs = squeeze(doppler_spectrum(rsn, :, :));
                    
                    % FFT频移对齐
                    dfs_shifted = fftshift(dfs, 1);
                    freq_bin_shifted = fftshift(freq_bin);
                    
                    % ==== 调试模式可打开 ====
                    figure('visible', 'off'); 
                    set(gcf, 'Position', [100, 100, 800, 600]); % 固定窗口大小
                    imagesc(dfs_shifted);
                    
                    % 隐藏所有坐标元素
                    axis off;
                    set(gca, 'Position', [0 0 1 1]);
                    
                    
                    % 保存图像
                    sname = [suname, '-', num2str(mn), '-', num2str(ln), '-', num2str(on), '-', num2str(rn), '-', num2str(rsn)];
                    saveas(gcf, [sapath, sname, '.jpg']);
                    close(gcf); % 关闭当前图像
                    
                    % ==== 打印成功信息 ====
                    disp(['Successfully saved: ', sname, '.jpg']);
                end
            end
        end
    end
end