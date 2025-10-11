% 文件路径和参数
dfs_path = 'D:\Widar3.0\T_R_DFS_weight\';
phase_path = 'D:\Widar3.0\STIFMM\';
sapath = 'D:\Widar3.0\DFSandPhase_images\';

% 确保输出目录存在
if ~exist(sapath, 'dir')
    mkdir(sapath);
end

% 遍历所有文件索引
for suname = 1
    for mn = 3
        for ln = 1
            for on = 3
                for rn = 1
                    % 生成文件名前缀
                    file_prefix = [num2str(suname), '-', num2str(mn), '-',...
                                  num2str(ln), '-', num2str(on), '-', num2str(rn)];
                    
                    % ==== 1. 加载数据 ====
                    dfs_file = fullfile(dfs_path, [file_prefix, '.mat']);
                    try
                        load(dfs_file, 'freq_weight_resized');
                        if ~isequal(size(freq_weight_resized), [224, 224])
                            error('DFS权重尺寸错误: 应为224x224, 实际为%s',...
                                 mat2str(size(freq_weight_resized)));
                        end
                    catch err
                        disp(['加载DFS权重失败: ', dfs_file, ' 错误: ', err.message]);
                        continue;
                    end
                    
                    % ==== 2. 处理相位图像 ====
                    phase_file = fullfile(phase_path, [file_prefix, '.jpg']);
                    phase_img = im2double(imread(phase_file));
                    phase_resized = imresize(phase_img, [224, 224], 'bilinear');
                    
                    % ==== 3. 转换到HSV空间 ====
                    phase_hsv = rgb2hsv(phase_resized);
                    
                    % ==== 4. 权重归一化与亮度映射 ====
                    % 归一化权重到[0.2, 1]范围 (保留最低亮度)
                    min_weight = min(freq_weight_resized(:));
                    max_weight = max(freq_weight_resized(:));
                    normalized_weight = (freq_weight_resized - min_weight) / (max_weight - min_weight + eps);
                    brightness_map = 0.2 + 0.8 * normalized_weight; % 亮度范围[0.2,1]
                    
                    % ==== 5. 调整HSV的V通道 ====
                    phase_hsv(:,:,3) = phase_hsv(:,:,3) .* brightness_map;
                    
                    % ==== 6. 转换回RGB并保存 ====
                    phase_weighted = hsv2rgb(phase_hsv);
                    phase_weighted = im2uint8(phase_weighted);
                    
                    save_name = fullfile(sapath, [file_prefix, '_weighted.jpg']);
                    imwrite(phase_weighted, save_name);
                    disp(['成功保存: ', save_name]);
                end
            end
        end
    end
end