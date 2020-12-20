def measure_br():
    global radar_config

    d_bin = int(DISTANCE//radar_config.range_resolution - 1)

    fs = radar_config.frame_rate 

    max_rpm = 35

    previous_fft_br = None

    N = radar_config.num_samples_per_chirp
    M = fs * BUFFER_TIME
    zero_amount = ZERO_PAD - M

    chirp_buffer = np.zeros((N//2, M), dtype=complex)
    dsp_buffer = np.zeros((N//2, M), dtype=complex)
    acorr_buffer = np.zeros((N//2, M//2 - 1))
    fft_buffer = np.zeros((N//2, ZERO_PAD//2), dtype=complex)

    fig, axs = plt.subplots(RANGE, 2)
    fig.set_figheight(6)
    fig.set_figwidth(12)

    save_num = 0
    chirps_collected = 0

    while(True):
        chirp_buffer = np.roll(chirp_buffer, -1, axis=1)
        chirp_buffer[:, M-1] = fft(fetch_json_packet(radar_socket)["data"])[0:N//2]
        chirps_collected += 1

        if chirps_collected < M:
            continue

        if chirps_collected == M:
            print("Buffer Filled!")

        if chirps_collected % radar_config.frame_rate != 0:
            continue

        # DSP
        for i in range(N//2):
            # Remove DC and Filter
            dsp_buffer[i] = detrend(chirp_buffer[i])
            dsp_buffer[i] = dsp.lp_filter_butterworth(dsp_buffer[i], cutoff=max_rpm/60, fs=fs, order=10)

            # Apply Hanning window and FFT of slow time signal
            sig_hanning = np.multiply(np.angle(dsp_buffer[i]), hann(M))
            sig_hanning = list(sig_hanning) + [0]*zero_amount
            fft_buffer[i] = fft(sig_hanning)[0:ZERO_PAD//2]

            # Apply autocorrelation 
            acorr = dsp.auto_correlate(np.angle(dsp_buffer[i]), fs)
            acorr_buffer[i] = dsp.lp_filter_butterworth(acorr, cutoff=3, fs=fs)
        
        # Clear Graphs
        for i in range(RANGE):
            axs[i, 0].cla()
            axs[i, 1].cla()
        
        # Save Matrix Data
        bin_range_buffer = chirp_buffer[d_bin - RANGE//2 : d_bin + RANGE//2 + 1]
        with open('buffer_{}.npy'.format(save_num), 'wb') as buffer_file:
            np.save(buffer_file, bin_range_buffer)
            save_num = (save_num + 1)%5

        # Check for BR at given distance
        predicted_brs = []
        predicted_brs_fft = []
        for i in range(d_bin - RANGE//2, d_bin + RANGE//2 + 1):
            acorr = acorr_buffer[i]
            lin_r = np.abs(linregress(np.arange(1, M // 2), acorr)[2])
            if lin_r > MAX_LIN_R:
                # predicted_brs.append("Too Linear {}".format(lin_r))
                continue
        
            # Mirror array to perform more accurate peak detection
            mirror_acorr = np.concatenate((acorr[::-1], acorr))
            max_samples_per_breath = fs * (60 // max_rpm)
            peaks = find_peaks(mirror_acorr, distance=max_samples_per_breath-1, prominence=0.4)[0]

            x = i-d_bin+1

            # Plotting FFT
            f_vals = np.linspace(0, fs/2, ZERO_PAD//2)[0:ZERO_PAD//32]
            st_fft = np.abs(fft_buffer[i][0:ZERO_PAD//32])
            st_fft = np.divide(st_fft, max(st_fft))
            min_peak_distance = max_rpm*ZERO_PAD/(60*fs)
            fft_peaks = find_peaks(st_fft, distance=min_peak_distance, prominence=0.3, height=0.7)[0]
            axs[x, 1].plot(f_vals, st_fft)
            if len(fft_peaks) > 0:
                axs[x, 1].plot(np.multiply(fft_peaks, fs/ZERO_PAD), st_fft[fft_peaks], "xr")

            # Plotting Autocorrelation
            axs[x, 0].plot(mirror_acorr)
            if (len(peaks) > 0):
                axs[x, 0].plot(peaks, mirror_acorr[peaks], 'xr')

            if (len(peaks) <= 2):
                # predicted_brs.append("Not Enough Peaks")
                continue

            # Calculate FFT Breathing Rate
            if len(fft_peaks) > 0:
                predicted_brs_fft.append(60*fft_peaks[0]*fs/ZERO_PAD)
                

            # Peak count is even meaning peak detection didn't detect harmonics
            # if (len(peaks) % 2 == 0):
            #     # predicted_brs.append("Even Peaks")
            #     continue

            # Idx should correspond to the first harmonic
            middle_idx = len(peaks) // 2
            # Always take first harmonic (lag > 1)
            idx_max_peak = peaks[middle_idx + 1]
            corr_val = mirror_acorr[idx_max_peak]
            # First peak is lag time
            lag = idx_max_peak + 1
            # Recenter peak index relative to acorr due to mirroring process
            lag -= len(mirror_acorr) // 2

            if corr_val < MIN_COR_VALUE:
                # predicted_brs.append("Corr Too Low")
                continue
            
            predicted_br = 1.25*60/(lag/fs)
            predicted_brs.append(predicted_br)
        
        plt.draw()
        plt.pause(0.000001)

        mean_br = 0
        if len(predicted_brs) > 0:
            mean_br = np.mean(np.array(predicted_brs))

        mean_br_fft = 0
        max_fft_br_std = 3
        if len(predicted_brs_fft) > 0:
            if previous_fft_br is not None or len(predicted_brs_fft) == RANGE:
                br_fft_sd = np.std(np.array(predicted_brs_fft))
                if previous_fft_br is None and br_fft_sd < max_fft_br_std:
                    mean_br_fft = np.mean(np.array(predicted_brs_fft))
                    previous_fft_br = mean_br_fft
                elif previous_fft_br is not None:
                    if br_fft_sd < max_fft_br_std:
                        mean_br_fft = np.mean(np.array(predicted_brs_fft))
                    else:
                        # find closest value to previous
                        mean_br_fft = min(predicted_brs_fft, key=lambda x:abs(x-previous_fft_br))

        if mean_br_fft != 0:
            previous_fft_br = mean_br_fft

        print("Auto BR: {}, FFT BR: {}".format( round(mean_br, 2), round(1.15*mean_br_fft, 2)))

        # res = radar_config.range_resolution
        # print_string = ""
        # for i in range(RANGE):
        #     print_string += "D {}: {}, ".format(round(DISTANCE + res*(i-RANGE//2), 1), predicted_brs[i])

        # print(print_string)


        # Autocorrelation