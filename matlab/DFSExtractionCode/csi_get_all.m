function [cfr_array, timestamp] = csi_get_all(filename)
csi_trace = read_bf_file(filename);
timestamp = zeros(length(csi_trace), 1);
cfr_array = zeros(length(csi_trace), 90);

valid_packet_count = 0;
for k = 1:length(csi_trace)
    csi_entry = csi_trace{k};
    
    % ========== Added error handling logic ==========
    if ~isstruct(csi_entry) || isempty(csi_entry)
        fprintf('Skipping invalid packet: file %s packet %d\n', filename, k);
        continue;  % Skip this packet
    end
    
    try
        csi_all = squeeze(get_scaled_csi(csi_entry)).';
        csi = [csi_all(:,1); csi_all(:,2); csi_all(:,3)].';
        timestamp(k) = csi_entry.timestamp_low;
        cfr_array(k,:) = csi;
        valid_packet_count = valid_packet_count + 1;
    catch
        fprintf('Processing failed: file %s packet %d\n', filename, k);
        cfr_array(k,:) = NaN;  % Mark invalid data
    end
end

% Trim invalid data
cfr_array = cfr_array(1:valid_packet_count, :);
timestamp = timestamp(1:valid_packet_count);
end