% READ_BF_FILE Reads beamforming feedback log files
%   This version uses the *C* version of the read_bfee function compiled as MATLAB MEX.
%
% (c) 2008-2011 Daniel Halperin <dhalperi@cs.washington.edu>
%
function ret = read_bf_file(filename)

%% Input validation
error(nargchk(1,1,nargin));  % Ensure exactly one parameter is passed

%% Open file
f = fopen(filename, 'rb');  % Open file in binary mode
if (f < 0)
    error('Couldn''t open file %s', filename);  % If file opening fails, throw error
    return;
end

status = fseek(f, 0, 'eof');  % Move file pointer to end of file
if status ~= 0
    [msg, errno] = ferror(f);  % Get error information
    error('Error %d seeking: %s', errno, msg);  % Throw error
    fclose(f);  % Close file
    return;
end
len = ftell(f);  % Get file length

status = fseek(f, 0, 'bof');  % Reset file pointer to beginning of file
if status ~= 0
    [msg, errno] = ferror(f);  % Get error information
    error('Error %d seeking: %s', errno, msg);  % Throw error
    fclose(f);  % Close file
    return;
end

%% Initialize variables
ret = cell(ceil(len/95),1);     % Initialize return value cell array, expecting 95 bytes per record
cur = 0;                        % Current file offset
count = 0;                      % Number of output records
broken_perm = 0;                % Flag for encountering corrupted CSI
triangle = [1 3 6];             % For antenna permutation sums can only be 1, 3, 6

%% Process all entries in the file, the entire file contains n bfee records, i.e., number of samples
% Need to read 3 bytes: 2-byte field_len size field and 1-byte code
% bfee = filed_len(2byte) + code(1byte) + field
while cur < (len - 3)
    % Read size and code
    field_len = fread(f, 1, 'uint16', 0, 'ieee-be');  % Read field length (big-endian format)
    code = fread(f, 1);  % Read code
    cur = cur + 3;  % Update current offset
    
    % If code is not 187, it's not channel information, skip this record and continue
    if (code == 187) % This represents channel information, get beamforming or physical data
        bytes = fread(f, field_len - 1, 'uint8=>uint8');  % Read data bytes to bytes
        cur = cur + field_len - 1;  % Update current offset
        if (length(bytes) ~= field_len - 1)  % Check the length of read bytes
           break;  % If length doesn't match, break the loop
        end
    else % Skip all other information
        fseek(f, field_len - 1, 'cof');  % filed_len = code + field, already read to filed
        % So at 'cof' (current position of file) only need to move back filed_len - 1
        cur = cur + field_len - 1;  % Update current offset
        continue;  % Continue to next loop
    end
    
    if (code == 187) % If it's beamforming matrix - output record
        count = count + 1;  % Increment record count

        ret{count} = read_bfee(bytes);  % Read CSI data
        
        perm = ret{count}.perm;  % Get permutation information
        Nrx = ret{count}.Nrx;  % Get number of receive antennas
        if Nrx == 1 % If only one antenna, no permutation needed
            continue;  % Continue to next loop
        end
        if sum(perm) ~= triangle(Nrx) % Check if matrix contains default values: depending on the number of receive antennas Nrx, which antenna receives data first has permutation
            % One antenna: 1; Two antennas: stored as 1,2 or 2,1; Three antennas: stored as 1,2,3 for example, so perm array sum = 1, 3, 6
            if broken_perm == 0
                broken_perm = 1;  % Mark as encountering corrupted permutation
                fprintf('WARN ONCE: Found CSI (%s) with Nrx=%d and invalid perm=[%s]\n', filename, Nrx, int2str(perm));
            end
        else
            % Update CSI data according to permutation
            ret{count}.csi(:,perm(1:Nrx),:) = ret{count}.csi(:,1:Nrx,:);  
        end
    end
end
ret = ret(1:count);  % Return only valid records

%% Close file
fclose(f);  % Close file
end
