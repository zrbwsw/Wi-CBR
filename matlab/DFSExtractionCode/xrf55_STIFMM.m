filepath = 'E:\XRF55\QFM\FME\';
sapath = 'E:\XRF55\QFM\STIFMM\';

% 创建输出目录
if ~exist(sapath, 'dir')
    mkdir(sapath);
end

% 遍历所有用户-动作-重复组合
for user = 35:39
    for action = 1:55
        for repeat = 1:20
            % 预加载第一个接收器图像
            base_name = [num2str(user), '-', num2str(action), '-', num2str(repeat)];
            first_receiver_path = fullfile(filepath, [base_name, '-1.jpg']);
            
            % 检查基础文件是否存在
            if ~exist(first_receiver_path, 'file')
                disp(['首文件缺失: ' base_name '-1.jpg']);
                continue;
            end
            
            % 初始化纵向拼接矩阵
            try
                sfm = imread(first_receiver_path);
                has_error = false;
            catch
                disp(['首文件读取失败: ' base_name '-1.jpg']);
                continue;
            end
            
            % 追加后续接收器图像
            for receiver = 2:3
                receiver_path = fullfile(filepath, [base_name, '-', num2str(receiver), '.jpg']);
                
                if ~exist(receiver_path, 'file')
                    disp(['接收器' num2str(receiver) '缺失: ' base_name]);
                    has_error = true;
                    break;
                end
                
                try
                    fm = imread(receiver_path);
                    sfm = [sfm; fm]; % 纵向拼接
                catch ME
                    disp(['接收器' num2str(receiver) '读取失败: ' ME.message]);
                    has_error = true;
                    break;
                end
            end
            
            % 保存有效结果
            if ~has_error
                output_path = fullfile(sapath, [base_name, '.jpg']);

            sfm = imresize(sfm, [563, 563]);
            imwrite(sfm, output_path);
            disp(['save',output_path,'success.']);
            end
        end
    end
end