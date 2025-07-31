% CSI 可视化：可视化每个 TR 对的数据，按照接收器合并所有时间，首先打开 untitled1.fig，然后运行此代码

% 设置原始 CSI 数据和输出图像的文件路径
filepath = 'D:\Widar3.0\CSI\20181205\user3\'; % 原始 CSI 数据路径
sapath = 'D:\Widar3.0\QFM\STIFMM_rsn\'; % 保存处理后图像的路径
uname = 'user3'; % 文件命名中的用户名
suname = '14'; % 保存文件的基础名称

% 循环遍历不同参数以读取和处理 CSI 数据
for mn = 1:6 % 第一参数（gestrue type）的循环
    for ln = 1:5 % 第二参数（tensor location）的循环
        for on = 1:5 % 第三参数（face orientation）的循环
            for rn = 1:5 % 第四参数（repetition number）的循环
                for rsn = 1:6 % 第五参数（Wi-Fi receiver id）的循环
                    mfm = zeros(1, 3); % 初始化一个1行3列的数组，存储每个天线的均值与方差比率,MIMO(多输入，多输出) 系统中的天线
                    % 构建当前数据集的文件名
                    filename = [filepath, uname, '-', num2str(mn), '-', num2str(ln), '-', num2str(on), '-', num2str(rn), '-r', num2str(rsn), '.dat'];
                    
                    % 从文件读取原始 CSI 数据
                    c1 = read_bf_file(filename);
                    dl = length(c1); % 获取数据长度
                    qfm = zeros(30, dl); % 初始化矩阵以存储处理后的数据
                    dt = 1; % 起始索引
                    k = 0; % 列索引的偏移
                    Num_subcarrier = 30; % 子载波数量
                    package = zeros(90, dl); % 初始化包以存储 CSI 值
                    %package_spatial = zeros(30, dl); % 单独提取出来三个接收天线维度
                    % 从数据中提取 CSI 跟踪
                    csi_trace = c1(dt:dl, 1);
                    for j2 = 1:3 % 遍历三个天线
                        for i = 1:length(c1) % 遍历所有 CSI 条目
                            row = 0; % 包中的行索引
                            csi_entry = csi_trace{i}; % 获取当前 CSI 条目
                            csi = get_scaled_csi(csi_entry); % 对 CSI 条目进行缩放
                            j1 = 1; % j1一直为一，因为只有一个发射器
                            for j3 = 1:Num_subcarrier % 遍历每个子载波
                                row = row + 1; % 行索引递增
                                % 将 CSI 值存储到包中
                                package((j2 - 1) * 30 + row, i - k) = csi(j1, j2, j3);
                            end
                        end
                        % 计算当前天线的均值与方差比率
                        package1 = abs(package((j2 - 1) * 30 + 1:(j2 - 1) * 30 + 30, :));
                        mf = mean(mean(package1) ./ var(package1)); % 均值-方差比率
                        mfm(1, j2) = mf; % 将比率存储到天线的数组中
                    end
                    % 找到均值与方差比率最大的天线和最小的天线
                    [~, nma] = max(mfm); % 最大比率的天线索引
                    [~, nmi] = min(mfm); % 最小比率的天线索引
                    
                    % 计算最大天线和最小天线之间的 CSI 比率
                    csiqdata = package((nma - 1) * 30 + 1:(nma - 1) * 30 + 30, :) ./ ...
                                package((nmi - 1) * 30 + 1:(nmi - 1) * 30 + 30, :); % CSI 比率
                    
                    % 从 CSI 比率中获取相位信息
                    qfm(:, :) = angle(csiqdata(:, :)); % 相位矩阵
                    
                    % 将相位矩阵可视化为图像
                    fmi = imagesc(qfm); % 从相位矩阵创建图像
                    set(gca, 'position', [0 0 1 1]); % 调整坐标轴位置
                    grid off; % 关闭网格
                    axis normal; % 设置坐标轴属性
                    axis off; % 隐藏坐标轴
                    set(gca, 'xtick', []); % 移除 x 轴刻度
                    set(gca, 'ytick', []); % 移除 y 轴刻度
                    
                    % 构建保存图像的文件名
                    sname = [suname, '-', num2str(mn), '-', num2str(ln), '-', num2str(on), '-', num2str(rn), '-r', num2str(rsn)];
                    saveas(fmi, strcat(sapath, sname, '.jpg')); % 保存图像为 .jpg
                    disp(['save', sname, 'success.']); % 显示成功信息
                end
            end
        end
    end
end
