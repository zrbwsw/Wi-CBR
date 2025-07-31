filepath = 'E:\XRF55\Scene_4\';
sapath = 'E:\XRF55\QFM\DFS_images\';

rx_cnt = 3;        % 接收器总数
rx_acnt = 3;       % 激活接收器数
method = 'stft';   % 频谱分析方法

% 创建输出目录
if ~exist(sapath,'dir')
    mkdir(sapath);
end

receivers = {'lb', 'lf', 'rb'}; % 接收器前缀
% ========== 用户配置修正版 ==========
user_config = struct(...
    'input_ids', {{'03', '04', '13'}}, ... % 注意双花括号包裹
    'output_base', 37   ...
);
% ========== 循环结构优化 ==========
for user_idx = 1:numel(user_config.input_ids) % 改用numel获取元素数量
    % 输入路径处理
    raw_user_id = user_config.input_ids{user_idx}; % 使用{}访问cell元素
    
    % 自动补零处理
    if length(raw_user_id) == 1
        user_folder = ['0', raw_user_id];
    else
        user_folder = raw_user_id;
    end
             
    % 输出编号映射（Scene_2起始为31）
    output_user_id = user_config.output_base + (user_idx - 1);
    
    % 动作循环
    for mn = 1:55
        mn_str = sprintf('%02d', mn);
        
        % 重复循环
        for rn = 1:20
            rn_str = sprintf('%02d', rn);
            
            % 接收器循环
            for rsn = 1:3
                % 构建完整路径
                receiver = receivers{rsn};
                user_dir = fullfile(filepath, receiver, user_folder);
                filename = fullfile(user_dir, ...
                    sprintf('%s_%s_%s.dat', user_folder, mn_str, rn_str));
                
                % 文件存在性检查
                if ~exist(filename, 'file')
                    fprintf('[跳过] 文件不存在: %s\n', filename);
                    continue;
                end
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
                sname = [num2str(output_user_id), '-', num2str(mn), '-', num2str(rn), '-', num2str(rsn)];
                saveas(fmi, fullfile(sapath, [sname, '.jpg']));
                disp(['保存成功: ', sname]);
                close(fmi);    % 关闭图形窗口
                delete(fmi);   % 删除图形对象
                clear fmi;     % 清除变量引用
            end
        end
    end
end