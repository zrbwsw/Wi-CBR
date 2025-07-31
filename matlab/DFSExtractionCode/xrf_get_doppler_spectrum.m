function [doppler_spectrum, freq_bin] = xrf_get_doppler_spectrum(filename, rx_cnt, rx_acnt, method)
% Set-Up Parameters
samp_rate = 200;
half_rate = samp_rate / 2;
uppe_orde = 6;
uppe_stop = 60;
lowe_orde = 3;
lowe_stop = 2;
[lu,ld] = butter(uppe_orde,uppe_stop/half_rate,'low');
[hu,hd] = butter(lowe_orde,lowe_stop/half_rate,'high');
freq_bins_unwrap = [0:samp_rate/2-1 -samp_rate/2:-1]'/samp_rate;
freq_lpf_sele = freq_bins_unwrap <= uppe_stop / samp_rate & freq_bins_unwrap >= -uppe_stop / samp_rate;
freq_lpf_positive_max = sum(freq_lpf_sele(2:length(freq_lpf_sele)/2));
freq_lpf_negative_min = sum(freq_lpf_sele(length(freq_lpf_sele)/2:end));

% ========== 关键修改 ==========
% 直接读取当前接收器文件
try
    [csi_data, ~] = csi_get_all(filename);
catch err
    error('文件读取失败: %s (%s)', filename, err.message);
end

% 初始化输出矩阵（单接收器三维结构）
doppler_spectrum = zeros(1, 1+freq_lpf_positive_max + freq_lpf_negative_min,...
    floor(size(csi_data, 1)));

% 数据预处理
csi_data = csi_data(round(1:1:size(csi_data,1)),:);

% 天线对选择逻辑
csi_mean = mean(abs(csi_data));
csi_var = sqrt(var(abs(csi_data)));
csi_mean_var_ratio = csi_mean./csi_var;
[~,idx] = max(mean(reshape(csi_mean_var_ratio,[30 rx_acnt]),1));

% 参考信号生成
csi_data_ref = repmat(csi_data(:,(idx-1)*30+1:idx*30), 1, rx_acnt);

% 幅度调整
csi_data_adj = zeros(size(csi_data));
csi_data_ref_adj = zeros(size(csi_data_ref));
alpha_sum = 0;
for jj = 1:30*rx_acnt
    amp = abs(csi_data(:,jj));
    alpha = min(amp(amp~=0));
    alpha_sum = alpha_sum + alpha;
    csi_data_adj(:,jj) = abs(abs(csi_data(:,jj))-alpha).*exp(1j*angle(csi_data(:,jj)));
end
beta = 1000*alpha_sum/(30*rx_acnt);
for jj = 1:30*rx_acnt
    csi_data_ref_adj(:,jj) = (abs(csi_data_ref(:,jj))+beta).*exp(1j*angle(csi_data_ref(:,jj)));
end

% 共轭相乘
conj_mult = csi_data_adj .* conj(csi_data_ref_adj);
conj_mult = [conj_mult(:,1:30*(idx - 1)) conj_mult(:,30*idx+1:90)];

% 滤波处理
for jj = 1:size(conj_mult, 2)
    conj_mult(:,jj) = filter(lu, ld, conj_mult(:,jj));
    conj_mult(:,jj) = filter(hu, hd, conj_mult(:,jj));
end

% PCA分析
pca_coef = pca(conj_mult);
conj_mult_pca = conj_mult * pca_coef(:,1);

% 时频分析
if strcmp(method, 'cwt')
    freq_time_prof_allfreq = scaled_cwt(conj_mult_pca,frq2scal(...
        [1:samp_rate/2 -1*samp_rate/2:-1],'cmor4-1',1/samp_rate),'cmor4-1');
elseif strcmp(method, 'stft')
    time_instance = 1:length(conj_mult_pca);
    window_size = round(samp_rate/4+1);
    if(~mod(window_size,2))
        window_size = window_size + 1;
    end
    freq_time_prof_allfreq = tfrsp(conj_mult_pca, time_instance,...
        samp_rate, tftb_window(window_size, 'gauss'));
end

% 频谱截取与归一化
freq_time_prof = freq_time_prof_allfreq(freq_lpf_sele, :);
freq_time_prof = abs(freq_time_prof) ./ repmat(sum(abs(freq_time_prof),1),...
    size(freq_time_prof,1), 1);

% 频率轴生成
freq_bin = [0:freq_lpf_positive_max -1*freq_lpf_negative_min:-1];

% 存储结果
if size(freq_time_prof,2) >= size(doppler_spectrum,3)
    doppler_spectrum(1,:,:) = freq_time_prof(:,1:size(doppler_spectrum,3));
else
    doppler_spectrum(1,:,:) = [freq_time_prof zeros(size(doppler_spectrum,2),...
        size(doppler_spectrum,3) - size(freq_time_prof,2))];
end
end