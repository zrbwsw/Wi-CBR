%GET_SCALED_CSI Converts a CSI struct to a channel matrix H.
%
% (c) 2008-2011 Daniel Halperin <dhalperi@cs.washington.edu>
%
function ret = get_scaled_csi(csi_st)
    % Extract CSI matrix from input CSI structure
    csi = csi_st.csi;

    % Calculate the squared magnitude of CSI (sum of squares of real and imaginary parts), representing the power of each element
    % conj(csi) returns the conjugate of CSI, csi .* conj(csi) is the squared magnitude of complex numbers
    csi_sq = csi .* conj(csi);
    
    % Sum the power of all elements to get the total power of the entire CSI matrix
    csi_pwr = sum(csi_sq(:));  % csi_sq(:) flattens the matrix to a vector, sum calculates the sum of the vector

    % Convert RSSI (Received Signal Strength Indicator) from dB to linear value (mW) and get total RSSI power
    rssi_pwr = dbinv(get_total_rss(csi_st));

    % Calculate the scaling ratio between CSI and signal power based on CSI power and RSSI power
    % The 30 here is the number of subcarriers, so we need to average the CSI power
    scale = rssi_pwr / (csi_pwr / 30);

    % Handle noise data, if no noise information (csi_st.noise == -127), set to default value -92 dB
    if (csi_st.noise == -127)
        noise_db = -92;  % Default noise value is -92 dB
    else
        noise_db = csi_st.noise;  % Use actual noise value
    end
    
    % Convert noise dB value to linear noise power (mW)
    thermal_noise_pwr = dbinv(noise_db);

    % Calculate quantization error power, assuming quantization error of +/- 1 for each CSI element
    % Nrx*Ntx is the number of receive antennas times transmit antennas, representing the number of elements per subcarrier
    % Quantization error power increases proportionally with Nrx and Ntx
    quant_error_pwr = scale * (csi_st.Nrx * csi_st.Ntx);

    % Calculate total noise and error power, including thermal noise and quantization error
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr;

    % Scale the CSI matrix to match actual channel power and convert to units of sqrt(SNR)
    % sqrt(scale / total_noise_pwr) is the scaling factor, considering signal power and noise power
    ret = csi * sqrt(scale / total_noise_pwr);

    % If there are 2 transmit antennas, further multiply the result by sqrt(2)
    % This is because two antennas bring additional gain
    if csi_st.Ntx == 2
        ret = ret * sqrt(2);
    % If there are 3 transmit antennas, use sqrt(3) for gain adjustment
    % Actually, this uses a 4.5dB approximation (sqrt(dbinv(4.5))), which is approximately equal to sqrt(3)
    elseif csi_st.Ntx == 3
        % sqrt(dbinv(4.5)) is an approximation, about 1.995
        % Actually, sqrt(3) is 1.732, but chip manufacturers often use 4.5 dB to simplify calculations
        ret = ret * sqrt(dbinv(4.5));
    end
end
