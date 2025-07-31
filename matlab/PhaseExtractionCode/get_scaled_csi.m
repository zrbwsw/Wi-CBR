%GET_SCALED_CSI Converts a CSI struct to a channel matrix H.
%
% (c) 2008-2011 Daniel Halperin <dhalperi@cs.washington.edu>
%
function ret = get_scaled_csi(csi_st)
    % 从输入的CSI结构体中提取出CSI矩阵
    csi = csi_st.csi;

    % 计算CSI的平方模（实部和虚部的平方和），表示每个元素的功率
    % conj(csi) 返回 CSI 的共轭，csi .* conj(csi) 是复数模的平方 
    csi_sq = csi .* conj(csi);
    
    % 对所有元素的功率进行求和，得到整个CSI矩阵的总功率
    csi_pwr = sum(csi_sq(:));  % csi_sq(:) 将矩阵展平为向量，sum 对向量求和

    % 将RSSI（接收信号强度）从dB转换为线性值（mW），并得到总的RSSI功率
    rssi_pwr = dbinv(get_total_rss(csi_st));

    % 根据CSI功率和RSSI功率计算出CSI和信号功率之间的缩放比例
    % 这里的 30 是子载波的数量，因此需要对CSI功率取平均
    scale = rssi_pwr / (csi_pwr / 30);

    % 处理噪声数据，如果没有噪声信息（csi_st.noise == -127），设置为默认值-92 dB
    if (csi_st.noise == -127)
        noise_db = -92;  % 默认噪声值为 -92 dB
    else
        noise_db = csi_st.noise;  % 使用实际的噪声值
    end
    
    % 将噪声的dB值转换为线性噪声功率（mW）
    thermal_noise_pwr = dbinv(noise_db);

    % 计算量化误差的功率，假设每个CSI元素的量化误差为 +/- 1
    % Nrx*Ntx 是接收天线数乘以发送天线数，表示每个子载波中的元素数量
    % 量化误差功率随Nrx和Ntx成比例增加
    quant_error_pwr = scale * (csi_st.Nrx * csi_st.Ntx);

    % 计算总的噪声和误差功率，包括热噪声和量化误差
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr;

    % 对CSI矩阵进行缩放，以匹配实际的信道功率，并将其转换为单位 sqrt(SNR)
    % sqrt(scale / total_noise_pwr) 是缩放因子，考虑了信号功率和噪声功率
    ret = csi * sqrt(scale / total_noise_pwr);

    % 如果有2个发送天线，则对结果进一步乘以 sqrt(2)
    % 这是因为两根天线会带来额外的增益
    if csi_st.Ntx == 2
        ret = ret * sqrt(2);
    % 如果有3个发送天线，使用 sqrt(3) 进行增益调整
    % 实际上，这里使用的是4.5dB的近似值（sqrt(dbinv(4.5)）），这大约等于 sqrt(3)
    elseif csi_st.Ntx == 3
        % sqrt(dbinv(4.5)) 是一个近似值，约为1.995
        % 实际上，sqrt(3) 是 1.732，但芯片制造商经常使用4.5 dB来简化计算
        ret = ret * sqrt(dbinv(4.5));
    end
end
