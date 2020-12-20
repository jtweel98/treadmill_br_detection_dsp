import socket
import sys
import signal
import time
import json
from radar_config import RadarConfig
from threading import Thread
from dsp import DigitalSignalProcessor
from scipy.fft import fft
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import detrend
from matplotlib.animation import FuncAnimation
from copy import deepcopy
import csv

# Constants
HEADER_LENGTH = 4
HOST = "192.168.0.168"

# Global Variables ------------------------------------------------
current_speed = 0
radar_socket = None
speed_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
speed_tracker_active = len(sys.argv) > 6
radar_config = None
dsp = None

# Helper Functions ------------------------------------------------
def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def fetch_json_packet(sock):
    packet_length = int(recvall(sock, HEADER_LENGTH).decode("utf-8"))
    serialized_packet = recvall(sock, packet_length).decode("utf-8")
    json_packet = json.loads(serialized_packet)
    return json_packet

# Signal Handler ------------------------------------------------
def signal_handler(sig, frame):
    radar_socket.close()
    speed_socket.close()
    exit(0)
signal.signal(signal.SIGINT, signal_handler)

# Thread Functions ------------------------------------------------
def speed_thread_func():
    while True:
        packet_length = int(recvall(speed_socket, HEADER_LENGTH).decode("utf-8"))
        current_speed = round(float(recvall(speed_socket, packet_length).decode("utf-8")), 2) # TODO: will need semaphore here

def run_session(title="Default", buffer_time=16, time_delay=0):
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

    time.sleep(time_delay)

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

# def new_func():
#     global radar_config, dsp
#     N = radar_config.num_samples_per_chirp
#     M = radar_config.frame_rate * 16

#     chirp_data = run_session()
#     fft_buffer = np.zeros((M, N//2), dtype=complex)

#     for i in range(M):
#         fft_buffer[i, :] = fft(detrend(chirp_data[i, :]))[0:N//2]
    
#     dsp.range_fft_gif(fft_buffer)


def single_fft_capture():
    global dsp
    chirp = fetch_json_packet(radar_socket)["data"]
    chirp = detrend(chirp)
    freq_data = fft(chirp)[0:len(chirp)//2]
    dsp.range_fft_plot(freq_data, "without dc")

def radar_main_thread():
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

if __name__ == "__main__":
    # Setup Socket Connections and Start Threads
    RADAR_PORT = 4242
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, RADAR_PORT))
    sock.listen(1)
    print("Listening on port {port} ...".format(port=RADAR_PORT))

    radar_socket, client_addr = sock.accept()
    print("Connected by: ", client_addr)

    radar_main_thread()



# OLD STUFF

# Collect Chirp


# fft_buffer = np.roll(fft_buffer, -1, axis=1)

# dsp.range_fft_bin_plot(fft_buffer, 0.6, type="abs", title_afix="_abs", br=True)

# # Remove DC from bin signals
# r_values = [0] * (N//2)
# for i in range(N//2):
#     fft_buffer[i] = detrend(fft_buffer[i])
#     corr_buffer[i], r_values[i], fft_buffer[i] = dsp.auto_correlation(fft_buffer[i], fs=radar_config.frame_rate)
#     # fft_buffer[i] = dsp.lp_filter_butterworth(sig=fft_buffer[i].real, cutoff=1.1, fs=radar_config.frame_rate)
#     # fft_buffer[i] = detrend(fft_buffer[i], type='linear')

# # Plot Range FFT
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# def animate(i):
    
# d_vals = np.linspace(0, radar_config.max_range, N//2)



# Plot real values for chirp within 5 range bins
# t_vals = np.linspace(0, buffer_time, M)
# corr_x_vals = np.linspace(0, M//2 - 1, M//2 - 1)
# test_d = 0.7
# bin_for_d = int(test_d//radar_config.range_resolution)
# fig, axs = plt.subplots(5)

# Plot Autocorrelation per bin
# for i in range(5):
#     bin = bin_for_d - 2 + i
#     axs[i].set_title("R Value: {r}".format(r=str(r_values[bin])))
#     print(str(r_values[bin]))
#     axs[i].plot(corr_x_vals, corr_buffer[bin])    

# Plot data per bin
# for i in range(5):
#     bin = bin_for_d - 2 + i
#     axs[i].set_title("Bin: {b}".format(b=str(bin)))
#     axs[i].plot(t_vals, fft_buffer[bin])

# plt.show()

# Cycle through bins
# time.sleep(3)
# fig = plt.figure(figsize=(20,10))
# # fig = plt.figure()
# ax = fig.add_subplot(111)
# Ln,  = ax.plot(np.real(fft_buffer[0]))
# plt.ion()
# plt.show()
# for i in range(N//2-16):
#     ax.set_title(str(radar_config.range_resolution*(i+1)))
#     data = np.real(fft_buffer[i])
#     Ln.set_ydata(data)
#     ax.set_ylim([min([min(data), -1]),max([max(data), 1])])
#     plt.pause(1)



# # collect data
# chirps_collected = 0
# while chirps_collected < M:
#     frame_data = radar.next_frame_data(raw=True)
#     first_chirp_sig = frame_data[0].tolist()
#     chirps_collected += 1

#     time_buffer = np.roll(time_buffer, -1, axis=1)
#     time_buffer[:, M-1] = first_chirp_sig

# # process chirps
# chirps_processed = 0

# while chirps_processed < M:
#     chirp = copy(time_buffer[:, chirps_processed])
#     chirp = dsp.decouple_dc(chirp)
#     chirp = dsp.filter_min_distance(chirp)
#     chirp_range_fft = fft(chirp)[0:N//2]
#     chirps_processed += 1

#     ftt_buffer = np.roll(ftt_buffer, -1, axis=1)
#     ftt_buffer[:, M-1] = chirp_range_fft

# t_vals = np.linspace(0, buffer_time, M)

# test_d = 0.8128
# fig, axs = plt.subplots(5)
# axs[0].plot(t_vals, ftt_buffer[int(test_d//radar.metrics.range_resolution) - 2].real)
# axs[1].plot(t_vals, ftt_buffer[int(test_d//radar.metrics.range_resolution) - 1].real)
# axs[2].plot(t_vals, ftt_buffer[int(test_d//radar.metrics.range_resolution) - 0].real)
# axs[3].plot(t_vals, ftt_buffer[int(test_d//radar.metrics.range_resolution) + 1].real)
# axs[4].plot(t_vals, ftt_buffer[int(test_d//radar.metrics.range_resolution) + 2].real)

# plt.plot(t_vals, ftt_buffer[0].real)
# plt.grid()
# plt.show()