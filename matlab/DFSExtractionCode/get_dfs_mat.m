% 定义全局参数
base_dir = 'D:\Widar3.0\CSI\20181211\20181211\user9';  % 原始数据根目录
output_dir = 'E:\DFS\16';  % 输出目录
rx_cnt = 6;                       % 接收器数量
rx_acnt = 3;                      % 有效天线对数量
method = 'stft';                  % 选择频谱分析方法: 'stft' 或 'cwt'
suname = 16;
% 确保输出目录存在
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
    fprintf('创建输出目录: %s\n', output_dir);
end

% 遍历所有实验配置
for id = 9                     % 用户ID范围
    for a = 1                  % 手势类型
        for b = 1                  % 躯干位置
            for c = 1           % 面部朝向
                for d = 1       % 重复次数
                    % 生成实验前缀
                    spfx_ges = sprintf('user%d-%d-%d-%d-%d', id, a, b, c, d);
                    fprintf('\n正在处理文件: %s\n', spfx_ges);
                    
                    try
                        % 调用多普勒频谱生成函数
                        [doppler_spectrum, freq_bin] = get_doppler_spectrum(...
                            fullfile(base_dir, spfx_ges), rx_cnt, rx_acnt, method);
                        
                        % 打印维度信息
                        fprintf('生成数据维度: %d×%d×%d (Rx×Freq×Time)\n',...
                            size(doppler_spectrum,1),...
                            size(doppler_spectrum,2),...
                            size(doppler_spectrum,3));
                        
                        % 保存为MATLAB格式
                        output_filename = fullfile(output_dir,...
                            sprintf('%d-%d-%d-%d-%d.mat', suname, a, b, c, d));
                        save(output_filename, 'doppler_spectrum', 'freq_bin', '-v7.3');
                        fprintf('已保存文件: %s\n', output_filename);
                        
                    catch ME  % 异常捕获
                        fprintf('!! 处理失败: %s\n错误信息: %s\n',...
                            spfx_ges, ME.message);
                        continue;
                    end
                end
            end
        end
    end
end