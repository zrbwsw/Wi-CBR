% 查看图像

% File paths and parameters
filepath = 'D:\Widar3.0\CSI\20181130_user5_10_11\user5\'; % Raw CSI path
sapath = 'D:\Widar3.0\DFS_images_look\'; % Save path for DFS images
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
                % Generate and save DFS image for each receiver
                for rsn = 1:rx_cnt
                    dfs = squeeze(doppler_spectrum(rsn, :, :));
                    
                    % FFT频移对齐
                    dfs_shifted = fftshift(dfs, 1);
                    freq_bin_shifted = fftshift(freq_bin);
                    
                    % ==== 调试模式：显示图像并暂停 ====
                    figure('visible', 'on'); % 设置为可见
                    set(gcf, 'Position', [100, 100, 800, 600]); % 固定窗口大小
                    imagesc(dfs_shifted);
                    

                    
                    % 频率轴标签
                    set(gca, 'YDir', 'normal'); % 确保Y轴从下到上递增
                    yticks = 1:20:length(freq_bin_shifted);
                    set(gca, 'YTick', yticks);
                    set(gca, 'YTickLabel', round(freq_bin_shifted(yticks)));
                    
                    % 时间轴标签 
                    time_steps = size(dfs_shifted, 2); % 修正变量名
                    xticks = 1:floor(time_steps/5):time_steps;
                    set(gca, 'XTick', xticks);
                    set(gca, 'XTickLabel', round(xticks / 1000, 1)); % 保留一位小数
                    
                    xlabel('Time (s)');
                    ylabel('Frequency Shift (Hz)');
                    title('DFS');
                    
                    % 保存图像前暂停，观察调试结果
                    pause(1); 
                    
                    % 直接保存高分辨率图像
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