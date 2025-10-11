/*
 * (c) 2008-2011 Daniel Halperin <dhalperi@cs.washington.edu>
 *
 * 该程序用于解析存储在 .dat 文件中的 CSI（信道状态信息）数据，
 * 并将其转换为 MATLAB 的结构体格式以便后续处理。
 * 该程序是基于 MATLAB MEX 文件的，用 C 语言编写。
 */

#include "mex.h"  // 包含 MATLAB 引擎接口函数

/* 核心的计算函数，用于解析输入字节并计算 CSI */
void read_bfee(unsigned char *inBytes, mxArray *outCell)
{
    // 解析时间戳（低 32 位），四个字节组成一个无符号长整型,timestamp_low 是指四个字节组成的时间戳的低 32 位，用于记录数据包的时间信息
    unsigned long timestamp_low = inBytes[0] + (inBytes[1] << 8) +
        (inBytes[2] << 16) + (inBytes[3] << 24);
    
    // 解析 beamforming 数据包计数器（2 字节）
    unsigned short bfee_count = inBytes[4] + (inBytes[5] << 8);
    
    // 解析接收天线数 Nrx 和发射天线数 Ntx
    unsigned int Nrx = inBytes[8];
    unsigned int Ntx = inBytes[9];
    
    // 解析 RSSI（接收信号强度指示）值，分别对应三根天线:信号的强度（RSSI）可以反映信号通过不同路径后的衰减
    unsigned int rssi_a = inBytes[10];
    unsigned int rssi_b = inBytes[11];
    unsigned int rssi_c = inBytes[12];
    
    // 解析噪声估计值
    char noise = inBytes[13];
    
    // 解析 AGC（自动增益控制）值:而 AGC 用来确保这些信号在接收器端被放大到合适的电平，以便后续的信道估计或处理算法可以正确执行
    unsigned int agc = inBytes[14];
    
    // 解析天线选择信息，用于判断 CSI 数据与实际天线的映射关系
    unsigned int antenna_sel = inBytes[15];
    
    // 解析 CSI 数据长度（2 字节）
    unsigned int len = inBytes[16] + (inBytes[17] << 8);
    
    // 解析速率及相关标志（2 字节）
    unsigned int fake_rate_n_flags = inBytes[18] + (inBytes[19] << 8);
    
    // 计算 CSI 数据长度的理论值，用于后续校验payload有效载荷长度len
    unsigned int calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 7) / 8; //+ 7 是为了确保在向下取整时正确计算字节（这是为了处理位数不整除的情况）
    
    unsigned int i, j;   // 用于循环遍历 CSI 数据的索引
    unsigned int index = 0, remainder;  // 用于 CSI 数据的位移操作
    unsigned char *payload = &inBytes[20];  // CSI 数据实际的有效载荷部分,跳过前20个
    char tmp;  // 临时变量用于存储位移操作后的 CSI 数据
    
    // 定义 CSI 数据的尺寸 [Ntx, Nrx, 30]，表示发射天线数、接收天线数和子载波数
    int size[] = {Ntx, Nrx, 30};
    
    // 在 MATLAB 中创建一个复数矩阵，用于存储 CSI 的实部和虚部
    mxArray *csi = mxCreateNumericArray(3, size, mxDOUBLE_CLASS, mxCOMPLEX); //三维数组，size定义每一维的大小，double,complex类型
    double* ptrR = (double *)mxGetPr(csi);  // 指向 CSI 实部的指针
    double* ptrI = (double *)mxGetPi(csi);  // 指向 CSI 虚部的指针

    // 检查实际的 CSI 数据长度是否与计算的理论长度匹配
    if (len != calc_len)
        mexErrMsgIdAndTxt("MIMOToolbox:read_bfee_new:size", "Wrong beamforming matrix size.");

    // 开始逐个解析 CSI 数据，遍历每个子载波（共 30 个）
    for (i = 0; i < 30; ++i)
    {
        index += 3;  // 跳过每个子载波的 3 字节数据（前 3 个字节无关）
        remainder = index % 8;  // 计算当前的位移

        // 遍历每对接收-发射天线组合
        for (j = 0; j < Nrx * Ntx; ++j)
        {
            // 提取 CSI 实部，使用位移操作获取相应的比特位
            tmp = (payload[index / 8] >> remainder) |
                  (payload[index/8+1] << (8-remainder));
            *ptrR = (double) tmp;  // 存储实部
            ++ptrR;  // 移动到下一个存储位置
            
            // 提取 CSI 虚部
            tmp = (payload[index / 8+1] >> remainder) |
                  (payload[index/8+2] << (8-remainder));
            *ptrI = (double) tmp;  // 存储虚部
            ++ptrI;  // 移动到下一个存储位置
            
            index += 16;  // 每次移动 16 位，表示 2 字节的数据（8 位 * 2）
        }
    }

    // 创建 MATLAB 数组，用于存储天线排列信息
    int perm_size[] = {1, 3};
    mxArray *perm = mxCreateNumericArray(2, perm_size, mxDOUBLE_CLASS, mxREAL);
    ptrR = (double *)mxGetPr(perm);
    
    // 解析天线选择信息，并将其存储在 perm 中，依次提取(1,2位;3,4位;5,6位)加一代表从1开始，
    //00表示第一个，01表示第二个，10表示第三个天线：加一变为1，2，3代表，该天线对应的是csi序列中第几个接收的
    ptrR[0] = ((antenna_sel) & 0x3) + 1;
    ptrR[1] = ((antenna_sel >> 2) & 0x3) + 1;
    ptrR[2] = ((antenna_sel >> 4) & 0x3) + 1;

    // 销毁之前在结构体中存储的旧字段，避免内存泄漏
    mxDestroyArray(mxGetField(outCell, 0, "timestamp_low"));
    mxDestroyArray(mxGetField(outCell, 0, "bfee_count"));
    mxDestroyArray(mxGetField(outCell, 0, "Nrx"));
    mxDestroyArray(mxGetField(outCell, 0, "Ntx"));
    mxDestroyArray(mxGetField(outCell, 0, "rssi_a"));
    mxDestroyArray(mxGetField(outCell, 0, "rssi_b"));
    mxDestroyArray(mxGetField(outCell, 0, "rssi_c"));
    mxDestroyArray(mxGetField(outCell, 0, "noise"));
    mxDestroyArray(mxGetField(outCell, 0, "agc"));
    mxDestroyArray(mxGetField(outCell, 0, "perm"));
    mxDestroyArray(mxGetField(outCell, 0, "rate"));
    mxDestroyArray(mxGetField(outCell, 0, "csi"));
    
    // 将解析出的各个数据字段重新赋值给结构体 outCell
    mxSetField(outCell, 0, "timestamp_low", mxCreateDoubleScalar((double)timestamp_low));
    mxSetField(outCell, 0, "bfee_count", mxCreateDoubleScalar((double)bfee_count));
    mxSetField(outCell, 0, "Nrx", mxCreateDoubleScalar((double)Nrx));
    mxSetField(outCell, 0, "Ntx", mxCreateDoubleScalar((double)Ntx));
    mxSetField(outCell, 0, "rssi_a", mxCreateDoubleScalar((double)rssi_a));
    mxSetField(outCell, 0, "rssi_b", mxCreateDoubleScalar((double)rssi_b));
    mxSetField(outCell, 0, "rssi_c", mxCreateDoubleScalar((double)rssi_c));
    mxSetField(outCell, 0, "noise", mxCreateDoubleScalar((double)noise));
    mxSetField(outCell, 0, "agc", mxCreateDoubleScalar((double)agc));
    mxSetField(outCell, 0, "perm", perm);
    mxSetField(outCell, 0, "rate", mxCreateDoubleScalar((double)fake_rate_n_flags));
    mxSetField(outCell, 0, "csi", csi);

    // 打印调试信息
    // printf("Nrx: %u Ntx: %u len: %u calc_len: %u\n", Nrx, Ntx, len, calc_len);
}

/* MEX 文件的主函数 这是 MEX 文件的标准接口，
nlhs 是输出参数的数量，plhs 是输出参数数组，
nrhs 是输入参数的数量，prhs 是输入参数数组。*/
void mexFunction(int nlhs, mxArray *plhs[],
	         int nrhs, const mxArray *prhs[])
{
    unsigned char *inBytes;  /* 输入的 CSI 数据字节流 */
    mxArray *outCell;        /* 输出的 MATLAB 结构体 */

    /* 检查输入和输出的参数数量是否正确 */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("MIMOToolbox:read_bfee_new:nrhs","One input required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MIMOToolbox:read_bfee_new:nlhs","One output required.");
    }

    /* 检查输入参数是否是 uint8 类型（即字节流） */
    if (!mxIsUint8(prhs[0])) {
        mexErrMsgIdAndTxt("MIMOToolbox:read_bfee_new:notBytes","Input must be a byte array");
    }

    /* 将输入的 MATLAB 数组转换为指向输入字节流的指针 */
    inBytes = (unsigned char *)mxGetData(prhs[0]);

    /* 创建一个 MATLAB 结构体，用于存储解析出的 CSI 数据 */
    const char *fieldnames[] = {"timestamp_low", "bfee_count", "Nrx", "Ntx",
                                "rssi_a", "rssi_b", "rssi_c", "noise",
                                "agc", "perm", "rate", "csi"};
    outCell = mxCreateStructMatrix(1, 1, 12, fieldnames);

    /* 调用核心函数解析输入的字节流 */
    read_bfee(inBytes, outCell);

    /* 设置输出结果 */
    plhs[0] = outCell;
}
