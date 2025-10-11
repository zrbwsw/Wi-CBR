% CSI data dimension optimization and timestamp statistics
filepath = 'D:\Widar3.0\QFM_MAT';  % Original data root directory
envs = 3;  % Environment identifier in old filenames

% Initialize dimension parameters
Num_subcarrier = 30;   % Number of subcarriers
Num_receivers = 6;     % Number of receive antennas

% ================== Storage initialization ==================
data_cell = {};        % New data storage cell
timestamps = [];       % Timestamp recording array
processed_files = 0;  % Successfully processed file counter

% ================== Five-dimensional parameter traversal ==================
for user_id = 13:16
    for gesture_id = 1:6
        for position_id = 1:5
            for orientation_id = 1:5
                for repeat_id = 1:5
                    % ================== Filename processing ==================
                    % Generate old and new filename pairs
                    old_filename = sprintf('%d-%d-%d-%d-%d-%d.mat',...
                        envs, user_id, gesture_id,...
                        position_id, orientation_id, repeat_id);
                    
                    new_filename = sprintf('%d-%d-%d-%d-%d.mat',...
                        user_id, gesture_id,...
                        position_id, orientation_id, repeat_id);
                    
                    % Construct file paths
                    user_folder = fullfile(filepath, num2str(user_id));
                    old_path = fullfile(user_folder, old_filename);
                    new_path = fullfile(user_folder, new_filename);

                    % ================== File operations ==================
                    if ~exist(old_path, 'file')
                        fprintf('[Missing] Old file: %s\n', old_filename);
                        continue;
                    end
                    
                    try
                        % ================== Data loading ==================
                        mat_data = load(old_path);
                        
                        % Data integrity verification
                        if ~isfield(mat_data, 'csi_data')
                            error('Missing csi_data field');
                        end
                        
                        % ================== Dimension processing ==================
                        % Original dimension validation
                        original_dims = size(mat_data.csi_data);
                        if ~isequal(original_dims(1:3), [Num_subcarrier, Num_receivers, 1])
                            error('Dimension anomaly: %s', mat2str(original_dims));
                        end
                        
                        % Compress third dimension (30×6×1×T → 30×6×T)
                        mat_data.csi_data = squeeze(mat_data.csi_data(:,:,1,:));
                        
                        % ================== File storage ==================
                        % Save optimized data to new filename
                        save(new_path, '-struct', 'mat_data');
                        
                        % ================== Data statistics ==================
                        current_t = size(mat_data.csi_data, 3);
                        timestamps(end+1) = current_t;
                        data_cell{end+1} = mat_data.csi_data;
                        processed_files = processed_files + 1;
                        
                        % Delete old file (Caution! Recommend commenting this line during testing)
                        % delete(old_path);  
                        
                        fprintf('[Success] %s → %s [%d×%d×%d]\n',...
                            old_filename, new_filename,...
                            size(mat_data.csi_data,1),...
                            size(mat_data.csi_data,2),...
                            current_t);
                        
                    catch ME
                        fprintf('[Failed] %s Error: %s\n',...
                            old_filename, ME.message);
                    end
                end
            end
        end
    end
end

% ================== Statistical report ==================
fprintf('\n===== Processing completed =====\n');
fprintf('Total processed files: %d\n', processed_files);
fprintf('Timestamp range: %d - %d\n', min(timestamps), max(timestamps));
fprintf('Dimension change: 30×6×1×T → 30×6×T\n');
fprintf('New file example: %s\\1\\1-1-1-1-1.mat\n', filepath);