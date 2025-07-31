% READ_BF_FILE 读取波束形成反馈日志文件。Reads in a file of beamforming feedback logs
%   此版本使用了编译为 MATLAB MEX 的 *C* 版本的 read_bfee 函数。
%
% (c) 2008-2011 Daniel Halperin <dhalperi@cs.washington.edu>
%
function ret = read_bf_file(filename)

%% 输入检查
error(nargchk(1,1,nargin));  % 确保传入的参数数量为1

%% 打开文件
f = fopen(filename, 'rb');  % 以二进制模式打开文件
if (f < 0)
    error('Couldn''t open file %s', filename);  % 如果文件打开失败，抛出错误
    return;
end

status = fseek(f, 0, 'eof');  % 将文件指针移动到文件末尾
if status ~= 0
    [msg, errno] = ferror(f);  % 获取错误信息
    error('Error %d seeking: %s', errno, msg);  % 抛出错误
    fclose(f);  % 关闭文件
    return;
end
len = ftell(f);  % 获取文件长度

status = fseek(f, 0, 'bof');  % 将文件指针重置到文件开头
if status ~= 0
    [msg, errno] = ferror(f);  % 获取错误信息
    error('Error %d seeking: %s', errno, msg);  % 抛出错误
    fclose(f);  % 关闭文件
    return;
end

%% 初始化变量
ret = cell(ceil(len/95),1);     % 初始化返回值单元格，预计每条记录95字节
cur = 0;                        % 当前文件偏移量
count = 0;                      % 输出记录数量
broken_perm = 0;                % 标记是否遇到损坏的 CSI
triangle = [1 3 6];             % 对于天线的排列求和只能是1，3，6

%% 处理文件中的所有条目，整个文件包含n个bfee记录，即采样个数
% 需要读取3个字节：2字节field_len大小字段和1字节代码code
% bfee = filed_len(2byte) + code(1byte) + field
while cur < (len - 3)
    % 读取大小和代码
    field_len = fread(f, 1, 'uint16', 0, 'ieee-be');  % 读取字段长度（大端格式）
    code = fread(f, 1);  % 读取代码
    cur = cur + 3;  % 更新当前偏移量
    
    % 如果代码非187，不是信道信息跳过该记录并继续
    if (code == 187) % 此时代表的是信道信息，获取波束形成或物理数据
        bytes = fread(f, field_len - 1, 'uint8=>uint8');  % 读取数据字节到bytes
        cur = cur + field_len - 1;  % 更新当前偏移量
        if (length(bytes) ~= field_len - 1)  % 检查读取的字节长度
           break;  % 如果长度不匹配，跳出循环
        end
    else % 跳过所有其他信息
        fseek(f, field_len - 1, 'cof');  % filed_len = code + field，当前已经读取到filed
        %所以在'cof'即current position of file 上仅仅需要后移 filed_len - 1
        cur = cur + field_len - 1;  % 更新当前偏移量
        continue;  % 继续下一个循环
    end
    
    if (code == 187) % 如果是波束形成矩阵 - 输出记录
        count = count + 1;  % 增加记录计数

        ret{count} = read_bfee(bytes);  % 读取 CSI 数据
        
        perm = ret{count}.perm;  % 获取排列信息
        Nrx = ret{count}.Nrx;  % 获取接收天线数量
        if Nrx == 1 % 如果只有一个天线，不需要排列
            continue;  % 继续下一个循环
        end
        if sum(perm) ~= triangle(Nrx) % 检查矩阵是否包含默认值:根据接收天线的数量Nrx的不同，先是第几个天线接收到数据有排列
            % 一个天线：1；两个天线：存储的是1，2或者2，1；三个天线：存储的是1，2，3例如，所以perm数组求和 = 1， 3， 6
            if broken_perm == 0
                broken_perm = 1;  % 标记为遇到损坏的排列
                fprintf('WARN ONCE: Found CSI (%s) with Nrx=%d and invalid perm=[%s]\n', filename, Nrx, int2str(perm));
            end
        else
            % 根据排列更新 CSI 数据
            ret{count}.csi(:,perm(1:Nrx),:) = ret{count}.csi(:,1:Nrx,:);  
        end
    end
end
ret = ret(1:count);  % 只返回有效记录

%% 关闭文件
fclose(f);  % 关闭文件
end
