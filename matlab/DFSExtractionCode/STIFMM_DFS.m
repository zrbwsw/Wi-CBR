filepath = 'E:\XRF55\QFM\DFS_images\';
sapath = 'E:\XRF55\QFM\STIFMM_DFS\';

% Create output directory
if ~exist(sapath, 'dir')
    mkdir(sapath);  
end

% Traverse all user-action-repeat combinations
for user = 1:39
    for action = 1:55
        for repeat = 1:20
            % Preload first receiver image
            base_name = [num2str(user), '-', num2str(action), '-', num2str(repeat)];
            first_receiver_path = fullfile(filepath, [base_name, '-1.jpg']);
            
            % Check if base file exists
            if ~exist(first_receiver_path, 'file')
                disp(['First file missing: ' base_name '-1.jpg']);
                continue;
            end
            
            % Initialize vertical concatenation matrix
            try
                sfm = imread(first_receiver_path);
                has_error = false;
            catch
                disp(['First file read failed: ' base_name '-1.jpg']);
                continue;
            end
            
            % Append subsequent receiver images
            for receiver = 2:3
                receiver_path = fullfile(filepath, [base_name, '-', num2str(receiver), '.jpg']);
                
                if ~exist(receiver_path, 'file')
                    disp(['Receiver' num2str(receiver) ' missing: ' base_name]);
                    has_error = true;
                    break;
                end
                
                try
                    fm = imread(receiver_path);
                    sfm = [sfm; fm]; % Vertical concatenation
                catch ME
                    disp(['Receiver' num2str(receiver) ' read failed: ' ME.message]);
                    has_error = true;
                    break;
                end
            end
            
            % Save valid results
            if ~has_error
                output_path = fullfile(sapath, [base_name, '.jpg']);
                sfm = imresize(sfm, [563, 563]);
                imwrite(sfm, output_path);
                disp(['save',output_path,'success.']);
            end
        end
    end
end