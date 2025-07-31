function [cfr_array, timestamp] = csi_get_all(filename)
csi_trace = read_bf_file(filename);
timestamp = zeros(length(csi_trace), 1);
cfr_array = zeros(length(csi_trace), 90);

valid_packet_count = 0;
for k = 1:length(csi_trace)
    csi_entry = csi_trace{k};
    
    % ========== 新增容错逻辑 ==========
    if ~isstruct(csi_entry) || isempty(csi_entry)
        fprintf('跳过无效数据包: 文件 %s 第 %d 包\n', filename, k);
        continue;  % 跳过此包
    end
    
    try
        csi_all = squeeze(get_scaled_csi(csi_entry)).';
        csi = [csi_all(:,1); csi_all(:,2); csi_all(:,3)].';
        timestamp(k) = csi_entry.timestamp_low;
        cfr_array(k,:) = csi;
        valid_packet_count = valid_packet_count + 1;
    catch
        fprintf('处理失败: 文件 %s 第 %d 包\n', filename, k);
        cfr_array(k,:) = NaN;  % 标记无效数据
    end
end

% 裁剪无效数据
cfr_array = cfr_array(1:valid_packet_count, :);
timestamp = timestamp(1:valid_packet_count);
end