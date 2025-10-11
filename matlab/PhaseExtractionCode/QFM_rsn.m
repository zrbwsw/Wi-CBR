% CSI Visualization: Visualize data for each TR pair, merge all time by receiver, first open untitled1.fig, then run this code

% Set file paths for original CSI data and output images
filepath = 'D:\Widar3.0\CSI\20181205\user3\'; % Original CSI data path
sapath = 'D:\Widar3.0\QFM\STIFMM_rsn\'; % Path to save processed images
uname = 'user3'; % Username in file naming
suname = '14'; % Base name for saved files

% Loop through different parameters to read and process CSI data
for mn = 1:6 % Loop for first parameter (gesture type)
    for ln = 1:5 % Loop for second parameter (tensor location)
        for on = 1:5 % Loop for third parameter (face orientation)
            for rn = 1:5 % Loop for fourth parameter (repetition number)
                for rsn = 1:6 % Loop for fifth parameter (Wi-Fi receiver id)
                    mfm = zeros(1, 3); % Initialize a 1x3 array to store mean-to-variance ratio for each antenna, antennas in MIMO (Multiple Input, Multiple Output) system
                    % Build filename for current dataset
                    filename = [filepath, uname, '-', num2str(mn), '-', num2str(ln), '-', num2str(on), '-', num2str(rn), '-r', num2str(rsn), '.dat'];
                    
                    % Read raw CSI data from file
                    c1 = read_bf_file(filename);
                    dl = length(c1); % Get data length
                    qfm = zeros(30, dl); % Initialize matrix to store processed data
                    dt = 1; % Starting index
                    k = 0; % Offset for column index
                    Num_subcarrier = 30; % Number of subcarriers
                    package = zeros(90, dl); % Initialize package to store CSI values
                    %package_spatial = zeros(30, dl); % Separately extract three receive antenna dimensions
                    % Extract CSI trace from data
                    csi_trace = c1(dt:dl, 1);
                    for j2 = 1:3 % Iterate through three antennas
                        for i = 1:length(c1) % Iterate through all CSI entries
                            row = 0; % Row index in package
                            csi_entry = csi_trace{i}; % Get current CSI entry
                            csi = get_scaled_csi(csi_entry); % Scale the CSI entry
                            j1 = 1; % j1 is always 1 because there's only one transmitter
                            for j3 = 1:Num_subcarrier % Iterate through each subcarrier
                                row = row + 1; % Increment row index
                                % Store CSI value in package
                                package((j2 - 1) * 30 + row, i - k) = csi(j1, j2, j3);
                            end
                        end
                        % Calculate mean-to-variance ratio for current antenna
                        package1 = abs(package((j2 - 1) * 30 + 1:(j2 - 1) * 30 + 30, :));
                        mf = mean(mean(package1) ./ var(package1)); % Mean-variance ratio
                        mfm(1, j2) = mf; % Store ratio in antenna array
                    end
                    % Find antenna with maximum and minimum mean-to-variance ratio
                    [~, nma] = max(mfm); % Index of antenna with maximum ratio
                    [~, nmi] = min(mfm); % Index of antenna with minimum ratio
                    
                    % Calculate CSI ratio between maximum and minimum antennas
                    csiqdata = package((nma - 1) * 30 + 1:(nma - 1) * 30 + 30, :) ./ ...
                                package((nmi - 1) * 30 + 1:(nmi - 1) * 30 + 30, :); % CSI ratio
                    
                    % Get phase information from CSI ratio
                    qfm(:, :) = angle(csiqdata(:, :)); % Phase matrix
                    
                    % Visualize phase matrix as image
                    fmi = imagesc(qfm); % Create image from phase matrix
                    set(gca, 'position', [0 0 1 1]); % Adjust axis position
                    grid off; % Turn off grid
                    axis normal; % Set axis properties
                    axis off; % Hide axes
                    set(gca, 'xtick', []); % Remove x-axis ticks
                    set(gca, 'ytick', []); % Remove y-axis ticks
                    
                    % Build filename for saving image
                    sname = [suname, '-', num2str(mn), '-', num2str(ln), '-', num2str(on), '-', num2str(rn), '-r', num2str(rsn)];
                    saveas(fmi, strcat(sapath, sname, '.jpg')); % Save image as .jpg
                    disp(['save', sname, 'success.']); % Display success message
                end
            end
        end
    end
end
