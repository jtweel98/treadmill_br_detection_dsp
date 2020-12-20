# Everything in here can be ignored for the most part

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

def run_session(title="Default", buffer_time=20, time_delay=0):
    global radar_config

    N = radar_config.num_samples_per_chirp
    M = radar_config.frame_rate * buffer_time

    # skip_time = 10
    # skip_samples = skip_time*radar_config.frame_rate

    # os.system('say "Skipping Samples"')
    # while skip_samples > 0:
    #     data_packet = fetch_json_packet(radar_socket)
    #     skip_samples -= 1
    # os.system('say "Done Skipping Samples"')

    chirp_buffer = np.zeros((M, N))

    chirps_collected = 0
    while chirps_collected < M:
        # Fetch packet
        data_packet = fetch_json_packet(radar_socket)
        
        # Collect chirp
        chirp = data_packet["data"]

        # Store chirp data
        chirp_buffer[chirps_collected] = chirp

        chirps_collected += 1

        print("Chirp Collection Progress: {}%".format((100*chirps_collected)//M), end="\r")
    return chirp_buffer

def fetch_static_clutter():
    data = []
    with open("static_clutter.csv", mode="r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        data = next(csv_reader)
        data = [ complex(val) for val in data ]
    
    return data

def make_giff():
    global radar_config

    # Load Config
    with open("config.json", "r") as file:
        config_packet = json.load(file)
        radar_config = RadarConfig(config_packet["data"])

    giff_fft = None
    giff_peaks = None
    with open('br_fft.npy', 'wb') as br_fft_file:
        with open('br_fft_peaks.npy', 'wb') as br_peaks_file:
            giff_fft = np.load(br_fft_file)
            giff_peaks = np.load(br_peaks_file)

    fs = radar_config.frame_rate
    f_vals = np.linspace(0, fs/2, ZERO_PAD//2)[0:ZERO_PAD//32]

    fig, ax = plt.subplots(1)
    ln1, = plt.plot([], [], 'ro')
    ln2, = plt.plot([], [], 'm*')

    def init():
        ax.set_xlim(0,1)
    
    def update(i):
        ln1.set_data(f_vals, giff_fft[i])
        
        peaks = giff_peaks[i]
        if peaks
        ln2.set_data(x, ycos)
    
    ani = FuncAnimation(

    )

def session_dsp(chirp_buffer, buffer_time, type="abs"):
    global radar_config, dsp

    N = radar_config.num_samples_per_chirp
    M = radar_config.frame_rate * buffer_time

    fft_buffer = np.zeros((M, N//2), dtype=complex)
    doppler_fft_buffer = np.zeros((N//2, M//2), dtype=complex)

    # static_clutter = fetch_static_clutter()

    # Compute range fft
    for i in range(M):
        chirp = deepcopy(chirp_buffer[i])
        chirp = detrend(chirp)
        fft_buffer[i] = fft(chirp)[0:N//2]
        # fft_buffer[i] = np.subtract(fft_buffer[i], static_clutter)
        print("Range FFT Progress: {}%".format((100*(i+1))//M), end="\r")

    # LPF Bin time series
    fr = radar_config.frame_rate
    for i in range(N//2):
        fft_buffer[:, i] = dsp.lp_filter_butterworth(fft_buffer[:, i], 4, fr)
    
    # Compute doppler fft
    for i in range(N//2):
        range_bin = deepcopy(fft_buffer[:, i])
        range_bin = detrend(range_bin)
        range_data = np.real(range_bin) if type=="real" else np.abs(range_bin)
        doppler_fft_buffer[i] = fft(range_data)[0:M//2]
        print("Doppler FFT Progress: {}%".format((100*(i+1))//(N//2)), end="\r")

    return fft_buffer, doppler_fft_buffer

def session_plots(range_fft_buffer, doppler_fft_buffer, title="Default", distance=0.6, skip_gif=True):
    global dsp

    # Subplots of FFT Bins Over Entire Buffer Time
    dsp.range_bin_plot(range_fft_buffer, distance, title_afix=title)

    # FFT Per Bin (doppler fft)
    dsp.doppler_fft_bin_plot(range_fft_buffer, distance, title_afix=title)

    if not skip_gif:
        # Range FFT GIF
        dsp.range_fft_gif(range_fft_buffer, title_afix=title)

def radar_main_thread_old():
    global radar_config, dsp

    # Fetch config data first
    config_packet = fetch_json_packet(radar_socket)
    assert config_packet["packet_type"] == "config"

    # Set global variables based on config
    radar_config = RadarConfig(config_packet["data"])
    dsp = DigitalSignalProcessor(radar_config)

    SESSION_DETAILS = {
        0: {
            "file_name": sys.argv[2],
            "complex_type": sys.argv[3],
            "distance": float(sys.argv[4]),
            "measurement_time": int(sys.argv[5])
        },
    }

    chirp_data = [None] * len(SESSION_DETAILS)
    range_fft_data = [None] * len(SESSION_DETAILS)
    doppler_fft_data = [None] * len(SESSION_DETAILS)

    for i in range(len(SESSION_DETAILS)):
        chirp_data[i] = run_session(
            buffer_time=SESSION_DETAILS[i]["measurement_time"], title=SESSION_DETAILS[i]["file_name"]
        )
    
    for i in range(len(SESSION_DETAILS)):
        range_fft_data[i], doppler_fft_data[i] = session_dsp(
            chirp_buffer=chirp_data[i], buffer_time=SESSION_DETAILS[i]["measurement_time"], type=SESSION_DETAILS[i]["complex_type"]
        )

    print("Plotting Data ...")
    for i in range(len(SESSION_DETAILS)):
        session_plots(range_fft_data[i], doppler_fft_data[i], SESSION_DETAILS[i]["file_name"], SESSION_DETAILS[i]["distance"], skip_gif=False)

    return