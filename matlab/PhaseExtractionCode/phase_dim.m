% CSI 数据维度优化和时间戳统计
filepath = 'D:\Widar3.0\QFM_MAT';  % 原始数据根目录
envs = 3;  % 旧文件名中的环境标识符

% 初始化维度参数
Num_subcarrier = 30;   % 子载波数量
Num_receivers = 6;     % 接收天线数量

% ================== 存储初始化 ==================
data_cell = {};        % 新数据存储单元
timestamps = [];       % 时间戳记录数组
processed_files = 0;  % 成功处理文件计数器

% ================== 五维参数遍历 ==================
for user_id = 13:16
    for gesture_id = 1:6
        for position_id = 1:5
            for orientation_id = 1:5
                for repeat_id = 1:5
                    % ================== 文件名处理 ==================
                    % 生成新旧文件名对
                    old_filename = sprintf('%d-%d-%d-%d-%d-%d.mat',...
                        envs, user_id, gesture_id,...
                        position_id, orientation_id, repeat_id);
                    
                    new_filename = sprintf('%d-%d-%d-%d-%d.mat',...
                        user_id, gesture_id,...
                        position_id, orientation_id, repeat_id);
                    
                    % 构造文件路径
                    user_folder = fullfile(filepath, num2str(user_id));
                    old_path = fullfile(user_folder, old_filename);
                    new_path = fullfile(user_folder, new_filename);

                    % ================== 文件操作 ==================
                    if ~exist(old_path, 'file')
                        fprintf('[缺失] 旧文件: %s\n', old_filename);
                        continue;
                    end
                    
                    try
                        % ================== 数据加载 ==================
                        mat_data = load(old_path);
                        
                        % 数据完整性校验
                        if ~isfield(mat_data, 'csi_data')
                            error('缺少csi_data字段');
                        end
                        
                        % ================== 维度处理 ==================
                        % 原始维度验证
                        original_dims = size(mat_data.csi_data);
                        if ~isequal(original_dims(1:3), [Num_subcarrier, Num_receivers, 1])
                            error('维度异常: %s', mat2str(original_dims));
                        end
                        
                        % 压缩第三维度 (30×6×1×T → 30×6×T)
                        mat_data.csi_data = squeeze(mat_data.csi_data(:,:,1,:));
                        
                        % ================== 文件存储 ==================
                        % 保存优化后的数据到新文件名
                        save(new_path, '-struct', 'mat_data');
                        
                        % ================== 数据统计 ==================
                        current_t = size(mat_data.csi_data, 3);
                        timestamps(end+1) = current_t;
                        data_cell{end+1} = mat_data.csi_data;
                        processed_files = processed_files + 1;
                        
                        % 删除旧文件 (谨慎操作！测试时建议注释此句)
                        % delete(old_path);  
                        
                        fprintf('[成功] %s → %s [%d×%d×%d]\n',...
                            old_filename, new_filename,...
                            size(mat_data.csi_data,1),...
                            size(mat_data.csi_data,2),...
                            current_t);
                        
                    catch ME
                        fprintf('[失败] %s 错误: %s\n',...
                            old_filename, ME.message);
                    end
                end
            end
        end
    end
end

% ================== 统计报告 ==================
fprintf('\n===== 处理完成 =====\n');
fprintf('总处理文件: %d\n', processed_files);
fprintf('时间戳范围: %d - %d\n', min(timestamps), max(timestamps));
fprintf('维度变化: 30×6×1×T → 30×6×T\n');
fprintf('新文件示例: %s\\1\\1-1-1-1-1.mat\n', filepath);