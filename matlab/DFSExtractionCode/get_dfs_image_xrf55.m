filepath = 'E:\XRF55\Scene_1\';
sapath = 'E:\XRF55\QFM\DFS_images\';

rx_cnt = 3;        % 接收器总数
rx_acnt = 3;       % 激活接收器数
method = 'stft';   % 频谱分析方法

% 创建输出目录
if ~exist(sapath,'dir')
    mkdir(sapath);
end

receivers = {'lb', 'lf', 'rb'}; % 接收器前缀

for username = 1:30 % 用户编号
    user_folder = sprintf('%02d', username); % 强制两位数用户目录
    
    for mn = 1:55 % 动作编号
        mn_str = sprintf('%02d', mn); % 两位数动作编号
        
        for rn = 1:20 % 重复编号
            rn_str = sprintf('%02d', rn);   % 两位数重复编号
            
            for rsn = 1:3 % 接收器循环
                % ========== 输入路径构建 ==========
                receiver = receivers{rsn};
                user_dir = fullfile(filepath, receiver, user_folder);
                 
                filename = fullfile(user_dir, [user_folder, '_', mn_str, '_', rn_str, '.dat']);
                
                try
                    % 调用修改后的处理函数
                    [doppler_spectrum, freq_bin] = xrf_get_doppler_spectrum(filename, rx_cnt, rx_acnt, method);
                catch err
                    disp(['Error processing ', filename, ': ', err.message]);
                    continue;
                end
                
                % ==== 频谱处理 ====
                dfs = squeeze(doppler_spectrum(1, :, :)); % 修改点1：使用固定维度索引
                dfs_shifted = fftshift(dfs, 1);
                
                % ==== 可视化保存 ====
                fmi = figure('visible', 'off');
                imagesc(dfs_shifted);
                axis off;
                set(gca, 'Position', [0 0 1 1]);
                
                % 生成输出文件名（保持原格式）
                sname = [num2str(username), '-', num2str(mn), '-', num2str(rn), '-', num2str(rsn)];
                saveas(fmi, fullfile(sapath, [sname, '.jpg']));
                disp(['保存成功: ', sname]);
                close(fmi);    % 关闭图形窗口
                delete(fmi);   % 删除图形对象
                clear fmi;     % 清除变量引用
            end
        end
    end
end