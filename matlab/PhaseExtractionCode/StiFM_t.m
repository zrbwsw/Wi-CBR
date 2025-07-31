% Combine the image of each TR pair 

filepath = 'D:\Widar3.0\QFM\FME\'; % Image file path of each TR pair
sapath = 'D:\Widar3.0\QFM\STIFMM_t\';
uname = '14';

for mn = 1:6
    for ln = 1:5
        for on = 1:5
            for rn = 1:5
                sfm = []; % Initialize combined image
                for segment = 1:30
                    % Build the filename for the first receiver
                    filename = [filepath, uname, '-', num2str(mn), '-', num2str(ln), '-', num2str(on), '-', num2str(rn), '-r', num2str(1), '_seg', num2str(segment), '.jpg'];                    
                    % Check if the file exists for the first receiver
                    if ~exist(filename, 'file')
                        break; % Exit the loop if the file is not found
                    end   
                    
                    % Read the first receiver's image
                    fm = imread(filename);
                    sfm = fm; % Initialize combined image as the first receiver's image                                                                                                  
                    
                    % Loop through other receivers (r2 to r6)
                    for rsn = 2:6
                        filename = [filepath, uname, '-', num2str(mn), '-', num2str(ln), '-', num2str(on), '-', num2str(rn), '-r', num2str(rsn), '_seg', num2str(segment), '.jpg'];
                        
                        % Check if the file exists for the current receiver
                        if exist(filename, 'file')
                            fm = imread(filename); % Read the image for the current receiver
                            sfm = [sfm; fm]; % Vertically concatenate images
                        else
                            break; % Exit if any receiver's image is missing
                        end
                    end
                    
                    % Save the combined image                
                    sname = [uname, '-', num2str(mn), '-', num2str(ln), '-', num2str(on), '-', num2str(rn), '-', num2str(segment), '.jpg'];
                    sfm = imresize(sfm, 0.28); % Resize the image
                    imshow(sfm);
                    imwrite(sfm, strcat(sapath, sname));
                    disp(['Saved ', sname, ' successfully.']);                                 
                end
            end
        end
    end
end
