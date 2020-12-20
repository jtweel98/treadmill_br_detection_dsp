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
import copy
import csv
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.signal.windows import hann
import cmath
import argparse

# Constants
HEADER_LENGTH = 4
HOST = "192.168.0.190"

C = 2.99792458e8
BUFFER_TIME = 30
DISTANCE = 1 # [m]
MAX_LIN_R = 0.5
MIN_COR_VALUE = 0.3
ZERO_PAD = 2**13

# Global Variables ------------------------------------------------
current_speed = 0
radar_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
speed_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
speed_tracker_active = False
radar_config = None
dsp = None
save_data = False

# Signal Handler ------------------------------------------------
def signal_handler(sig, frame):
    make_giff()
    radar_socket.close()
    speed_socket.close()
    exit(0)
signal.signal(signal.SIGINT, signal_handler)

# Helper Functions -------------------------------------------------
def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def perc_error(data, ref):
    data = np.abs(np.subtract(data, ref))
    perc_error = np.divide(data, ref)
    return 100*np.mean(perc_error)

def fetch_json_packet(sock):
    packet_length = int(recvall(sock, HEADER_LENGTH).decode("utf-8"))
    serialized_packet = recvall(sock, packet_length).decode("utf-8")
    json_packet = json.loads(serialized_packet)
    return json_packet

def speed_to_hz(speed):
    # from lin regression in matlab
    return 0.558818761181517 + 0.328498003233751*speed

def phase_unwrap(phase_buffer, remove_dc=True):
    for j in range(len(phase_buffer)):
        angle = phase_buffer[j]
        if j > 0:
            prev_angle = phase_buffer[j-1]
            if angle - prev_angle > np.pi:
                angle = angle - 2*np.pi
            elif angle - prev_angle < -np.pi:
                angle = angle + 2*np.pi
        phase_buffer[j] = angle
    phase_buffer = detrend(phase_buffer) if remove_dc else phase_buffer
    return phase_buffer

def get_phase_array(range_fft, speed=0):
    global dsp, radar_config, current_speed

    fs = radar_config.frame_rate

    high_hz = 2
    low_hz = 0.15
    if speed_tracker_active or speed != 0:
        high_hz = speed_to_hz(speed) if speed !=0 else speed_to_hz(current_speed)
        low_hz = 0.65

    phase_array = np.angle(range_fft)
    phase_array = phase_unwrap(phase_array)
    phase_array = detrend(phase_array)
    phase_array = dsp.bp_filter_butterworth(phase_array, high_cutoff=high_hz, low_cutoff=low_hz, fs=fs, order=10)
    return phase_array

def configure_radar():
    global radar_config, dsp

    # Fetch config data first
    config_packet = fetch_json_packet(radar_socket)
    assert config_packet["packet_type"] == "config"

    # Set global variables based on config
    radar_config = RadarConfig(config_packet["data"])
    dsp = DigitalSignalProcessor(radar_config)

    # Save to File
    if save_data:
        with open("config.json", "w") as file:
            json.dump(config_packet, file)

def single_fft_capture():
    global dsp
    chirp = fetch_json_packet(radar_socket)["data"]
    chirp = detrend(chirp)
    freq_data = fft(chirp)[0:len(chirp)//2]
    dsp.range_fft_plot(freq_data, "without dc")

# Plotting and Processing Functions -------------------------------------------------
def offline_processing(file_number, br):
    global radar_config, dsp

    # Load Config
    with open("config.json", "r") as file:
        config_packet = json.load(file)
        radar_config = RadarConfig(config_packet["data"])
        dsp = DigitalSignalProcessor(radar_config)

    fs = radar_config.frame_rate 
    N = radar_config.num_samples_per_chirp
    M = fs * BUFFER_TIME

    zero_amount = ZERO_PAD - M
    max_rpm = 35

    buffer = None
    br_data_fft = None
    br_data_acorr = None

    fig1, axs1 = plt.subplots(1, 1)
    fig2, axs2 = plt.subplots(1, 1)
    fig3, axs3 = plt.subplots(1, 1)
    fig4, axs4 = plt.subplots(2)

    with open('buffer_{}.npy'.format(file_number), 'rb') as buffer_file:
        buffer = np.load(buffer_file)

    with open('speed_{}.npy'.format(file_number), 'rb') as speed_file:
        speed = np.load(speed_file)[0]

    with open('br_fft.npy', 'rb') as buffer_file:
        br_data_fft = np.load(buffer_file)

    with open('br_acorr.npy', 'rb') as buffer_file:
        br_data_acorr = np.load(buffer_file)

    # Remove DC and Filter
    phase_array = get_phase_array(buffer, speed)

    # Apply Hanning window and FFT of slow time signal
    sig_hanning = np.multiply(phase_array, hann(M))
    sig_hanning = list(sig_hanning) + [0]*zero_amount
    st_fft = fft(sig_hanning)[0:ZERO_PAD//2]
    st_fft = np.abs(st_fft[0:ZERO_PAD//32])
    f_vals = np.linspace(0, fs/2, ZERO_PAD//2)[0:ZERO_PAD//32]
    min_peak_distance = max_rpm*ZERO_PAD/(60*fs)
    fft_peaks = find_peaks(st_fft, distance=min_peak_distance, prominence=0.3, height=0.7)[0]
    axs2.plot(f_vals, st_fft)
    if len(fft_peaks) > 0:
        axs2.plot(np.multiply(fft_peaks, fs/ZERO_PAD), st_fft[fft_peaks], "xr")

    # Apply autocorrelation 
    acorr = dsp.auto_correlate(phase_array, fs)
    max_samples_per_breath = fs * (60 // max_rpm)
    peaks = find_peaks(acorr, distance=max_samples_per_breath-1, prominence=0.4)[0]

    axs1.plot(acorr)
    if (len(peaks) > 0):
            axs1.plot(peaks, acorr[peaks], 'xr')

    axs3.plot(phase_array)

    axs4[0].plot(br_data_fft)
    axs4[0].plot([br]*len(br_data_fft), "g--")
    axs4[1].plot(br_data_acorr)
    axs4[1].plot([br]*len(br_data_fft), "g--")

    plt.show()

def plot_results():
    global radar_config, dsp

    # Load Config
    with open("config.json", "r") as file:
        config_packet = json.load(file)
        radar_config = RadarConfig(config_packet["data"])
        dsp = DigitalSignalProcessor(radar_config)

    fs = radar_config.frame_rate 
    N = radar_config.num_samples_per_chirp
    M = fs * BUFFER_TIME

    zero_amount = ZERO_PAD - M
    max_rpm = 35

    buffer = None
    br_data_fft = None
    br_data_acorr = None

    fig1, axs1 = plt.subplots(1, 2)
    fig2, axs2 = plt.subplots(1, 2)
    fig3, axs3 = plt.subplots(1, 2)
    fig4, axs4 = plt.subplots(1, 2)
    fig5, axs5 = plt.subplots(1, 2)
    fig6, axs6 = plt.subplots(1, 2)
    fig7, axs7 = plt.subplots(1)

    # Plot Slow Time Before & After Phase Unwrapping -------------------------------------------------------------
    slow_time_signal = None
    with open('./Part 1 Results (20BPM)/buffer_7.npy', 'rb') as buffer_file:
        slow_time_signal = np.load(buffer_file)
    slow_time_signal = slow_time_signal[0:525]
    slow_time_signal_after = copy.copy(slow_time_signal)
    x_vals = np.linspace(0, 1.17*len(slow_time_signal)/fs, len(slow_time_signal))
    slow_time_signal = np.angle(slow_time_signal)
    slow_time_signal_after = np.angle(slow_time_signal_after)
    slow_time_signal_after = phase_unwrap(slow_time_signal_after)
    slow_time_signal_after = dsp.lp_filter_butterworth(slow_time_signal_after, cutoff=1, fs=fs)
    axs1[0].plot(x_vals, slow_time_signal, "b-")
    axs1[0].plot(x_vals, [np.pi]*len(x_vals), "r--")
    axs1[0].plot(x_vals, [-np.pi]*len(x_vals), "r--")

    wavelength = C/radar_config.center_frequency

    axs1[1].plot(x_vals, np.multiply(slow_time_signal_after, 1000*wavelength/(2*np.pi)), "b-")
    axs1[0].set(
        title="Slow Time Signal Before Phase Unwrapping",
        ylabel="Beat Signal Angle (rads)",
        xlabel="Time (s)"
    )
    axs1[1].set(
        title="Slow-Time Signal After Phase Unwrapping",
        ylabel="Chest Displacement (mm)",
        xlabel="Time (s)"
    )
    fig1.set_figheight(4)
    fig1.set_figwidth(16)
    fig1.savefig("./Result Plots/STS Before and After.png", dpi=200, bbox_inches='tight')

    # Plot Showing Autocorrelation and FFT -------------------------------------------------------------
    with open('./Part 1 Results (20BPM)/buffer_7.npy', 'rb') as buffer_file:
        slow_time_signal = np.load(buffer_file)
    # Remove DC and Filter
    phase_array = get_phase_array(slow_time_signal, 0)
    # Apply Hanning window and FFT of slow time signal
    sig_hanning = np.multiply(phase_array, hann(len(phase_array)))
    sig_hanning = list(sig_hanning) + [0]*zero_amount
    st_fft = fft(sig_hanning)[0:ZERO_PAD//2]
    st_fft = np.abs(st_fft[0:ZERO_PAD//32])
    st_fft = np.divide(st_fft, max(st_fft))
    f_vals = np.linspace(0, 1.17*fs/2, ZERO_PAD//2)[0:ZERO_PAD//32]
    min_peak_distance = max_rpm*ZERO_PAD/(60*fs)
    fft_peaks = find_peaks(st_fft, distance=min_peak_distance, prominence=0.3, height=0.7)[0]
    # Apply autocorrelation 
    acorr = dsp.auto_correlate(phase_array, fs, offline_plot=True)
    max_samples_per_breath = fs * (60 // max_rpm)
    peaks = find_peaks(acorr, distance=max_samples_per_breath-1, prominence=0.4)[0]
    t_vals = np.linspace(0, len(acorr)/(1.17*fs), len(acorr))
    axs2[0].plot(t_vals, acorr, "b-")
    if len(peaks) > 0:
        axs2[0].plot(np.divide(peaks, 1.17*fs), acorr[peaks], "xr")
    axs2[1].plot(f_vals, st_fft, "b-")
    if len(fft_peaks) > 0:
        axs2[1].plot(np.multiply(fft_peaks, 1.17*fs/ZERO_PAD), st_fft[fft_peaks], "xr")

    axs2[0].set(
        title="Autocorrelated Slow-Time Signal",
        ylabel="Normalized Correlation Coefficient",
        xlabel="Delay Time (s)"
    )
    axs2[1].set(
        title="Doppler FFT",
        ylabel="Normalized Magnitude",
        xlabel="Frequency (Hz)"
    )
    fig2.set_figheight(4)
    fig2.set_figwidth(16)
    fig2.savefig("./Result Plots/Acorr of STS and Doppler FFT.png", dpi=200, bbox_inches='tight')

    # Plot Showing Autocorrelation and FFT -------------------------------------------------------------
    acorr_sit_20_bpm = None
    fft_sit_20_bpm = None
    acorr_sit_15_bpm = None
    fft_sit_15_bpm = None
    acorr_stand_20_bpm = None
    fft_stand_20_bpm = None
    acorr_stand_15_bpm = None
    fft_stand_15_bpm = None
    with open('./Part 2 Results (sit 20BPM)/br_acorr.npy', 'rb') as buffer_file:
        acorr_sit_20_bpm = np.load(buffer_file)
    with open('./Part 2 Results (sit 20BPM)/br_fft.npy', 'rb') as buffer_file:
        fft_sit_20_bpm = np.load(buffer_file)
    with open('./Part 2 Results (sit 15BPM)/br_acorr.npy', 'rb') as buffer_file:
        acorr_sit_15_bpm = np.load(buffer_file)
    with open('./Part 2 Results (sit 15BPM)/br_fft.npy', 'rb') as buffer_file:
        fft_sit_15_bpm = np.load(buffer_file)
    with open('./Part 2 Results (stand 20BPM)/br_acorr.npy', 'rb') as buffer_file:
        acorr_stand_20_bpm = np.load(buffer_file)
    with open('./Part 2 Results (stand 20BPM)/br_fft.npy', 'rb') as buffer_file:
        fft_stand_20_bpm = np.load(buffer_file)
    with open('./Part 2 Results (stand 15BPM)/br_acorr.npy', 'rb') as buffer_file:
        acorr_stand_15_bpm = np.load(buffer_file)
    with open('./Part 2 Results (stand 15BPM)/br_fft.npy', 'rb') as buffer_file:
        fft_stand_15_bpm = np.load(buffer_file)

    axs3[0].plot(np.multiply(acorr_sit_20_bpm[20:81], 1.1), "b-", label='Sitting')
    axs3[0].plot(np.multiply(acorr_stand_20_bpm[0:61], 1.1), "g-", label='Standing')
    axs3[0].plot([20]*60, 'k--', linewidth=1.5, label='Metronome Reference')

    axs3[0].set(
        title="Autocorrelation Breathing Rate Estimation (20 BPM)",
        ylabel="Breathing Rate (BPM)",
        xlabel="Time (s)"
    )
    axs3[0].set_ylim([17, 23])

    print("Acorr Sitting 20BPM - Error: ", perc_error(np.multiply(acorr_sit_20_bpm[20:81], 1.1), 20))
    print("Acorr Standing 20BPM - Error: ", perc_error(np.multiply(acorr_stand_20_bpm[0:61], 1.1), 20))

    axs3[1].plot(fft_sit_20_bpm[20:81], "b-", label='Sitting')
    axs3[1].plot(fft_stand_20_bpm[0:61], "g-", label='Standing')
    axs3[1].plot([20]*60, 'k--', linewidth=1.5, label='Metronome Reference')

    axs3[1].set(
        title="Doppler FFT Breathing Rate Estimation (20 BPM)",
        ylabel="Breathing Rate (BPM)",
        xlabel="Time (s)"
    )
    axs3[1].set_ylim([17, 23])

    print("FFT Sitting 20BPM - Error: ", perc_error(fft_sit_20_bpm[20:81], 20))
    print("FFT Standing 20BPM - Error: ", perc_error(fft_stand_20_bpm[0:61], 20))

    axs3[0].legend()
    axs3[1].legend()

    fig3.set_figheight(4)
    fig3.set_figwidth(16)
    fig3.savefig("./Result Plots/Standing and Sitting at 20 BPM.png", dpi=200, bbox_inches='tight')

    axs4[0].plot(np.multiply(acorr_sit_15_bpm[0:61], 1.1), "b-", label='Sitting')
    axs4[0].plot(np.multiply(acorr_stand_15_bpm[0:61], 1.1), "g-", label='Standing')
    axs4[0].plot([15]*60, 'k--', linewidth=1.5, label='Metronome Reference')

    axs4[0].set(
        title="Autocorrelation Breathing Rate Estimation (15 BPM)",
        ylabel="Breathing Rate (BPM)",
        xlabel="Time (s)"
    )
    axs4[0].set_ylim([12, 18])

    print("Acorr Sitting 15BPM - Error: ", perc_error(np.multiply(acorr_sit_15_bpm[0:61], 1.1), 15))
    print("Acorr Standing 15BPM - Error: ", perc_error(np.multiply(acorr_stand_15_bpm[0:61], 1.1), 15))

    axs4[1].plot(fft_sit_15_bpm[110:171], "b-", label='Sitting')
    axs4[1].plot(fft_stand_15_bpm[0:61], "g-", label='Standing')
    axs4[1].plot([15]*60, 'k--', linewidth=1.5, label='Metronome Reference')

    axs4[1].set(
        title="Doppler FFT Breathing Rate Estimation (15 BPM)",
        ylabel="Breathing Rate (BPM)",
        xlabel="Time (s)"
    )
    axs4[1].set_ylim([12, 18])

    print("FFT Sitting 15BPM - Error: ", perc_error(fft_sit_15_bpm[110:171], 15))
    print("FFT Standing 15BPM - Error: ", perc_error(fft_stand_15_bpm[0:61], 15))

    axs4[0].legend()
    axs4[1].legend()

    fig4.set_figheight(4)
    fig4.set_figwidth(16)
    fig4.savefig("./Result Plots/Standing and Sitting at 15 BPM.png", dpi=200, bbox_inches='tight')

    # Plot with Walking and Running -------------------------------------------------------------
    acorr_walk_55_bpm = None
    fft_walk_55_bpm = None
    acorr_walk_50_bpm = None
    fft_walk_50_bpm = None
    acorr_run_55_bpm = None
    fft_run_55_bpm = None
    acorr_run_50_bpm = None
    fft_run_50_bpm = None
    with open('./Part 3 Results (walk 55BPM)/br_acorr.npy', 'rb') as buffer_file:
        acorr_walk_55_bpm = np.load(buffer_file)
    with open('./Part 3 Results (walk 55BPM)/br_fft.npy', 'rb') as buffer_file:
        fft_walk_55_bpm = np.load(buffer_file)
    with open('./Part 3 Results (walk 50BPM)/br_acorr.npy', 'rb') as buffer_file:
        acorr_walk_50_bpm = np.load(buffer_file)
    with open('./Part 3 Results (walk 50BPM)/br_fft.npy', 'rb') as buffer_file:
        fft_walk_50_bpm = np.load(buffer_file)
    with open('./Part 3 Results (run 55BPM)/br_acorr.npy', 'rb') as buffer_file:
        acorr_run_55_bpm = np.load(buffer_file)
    with open('./Part 3 Results (run 55BPM)/br_fft.npy', 'rb') as buffer_file:
        fft_run_55_bpm = np.load(buffer_file)
    with open('./Part 3 Results (run 50BPM)/br_acorr.npy', 'rb') as buffer_file:
        acorr_run_50_bpm = np.load(buffer_file)
    with open('./Part 3 Results (run 50BPM)/br_fft.npy', 'rb') as buffer_file:
        fft_run_50_bpm = np.load(buffer_file)

    axs5[0].plot(acorr_walk_55_bpm[0:61], "b-", label='Brisk Walk (5.25 km/h)')
    axs5[0].plot(acorr_run_55_bpm[0:61], "g-", label='Running (9.00 km/h)')
    axs5[0].plot([55]*60, 'k--', linewidth=1.5, label='Metronome Reference')

    axs5[0].set(
        title="Autocorrelation Breathing Rate Estimation (55 BPM)",
        ylabel="Breathing Rate (BPM)",
        xlabel="Time (s)"
    )
    axs5[0].set_ylim([50, 60])

    print("Acorr Walk 55BPM - Error: ", perc_error(acorr_walk_55_bpm[0:61], 55))
    print("Acorr Run 55BPM - Error: ", perc_error(acorr_run_55_bpm[0:61], 55))

    axs5[1].plot(fft_walk_55_bpm[15:46], "b-", label='Brisk Walk (5.25 km/h)')
    axs5[1].plot(fft_run_55_bpm[15:46], "g-", label='Running (9.00 km/h)')
    axs5[1].plot([55]*60, 'k--', linewidth=1.5, label='Metronome Reference')

    axs5[1].set(
        title="Doppler FFT Breathing Rate Estimation (55 BPM)",
        ylabel="Breathing Rate (BPM)",
        xlabel="Time (s)"
    )
    # axs5[1].set_ylim([50, 60])

    print("FFT Walk 55BPM - Error: ", perc_error(fft_walk_55_bpm[15:46], 55))
    print("FFT Run 55BPM - Error: ", perc_error(fft_run_55_bpm[15:46], 55))

    axs5[0].legend()
    axs5[1].legend()

    fig5.set_figheight(4)
    fig5.set_figwidth(16)
    fig5.savefig("./Result Plots/Walking and Running at 55 BPM.png", dpi=200, bbox_inches='tight')

    axs6[0].plot(acorr_walk_50_bpm[0:31], "b-", label='Brisk Walk (5.25 km/h)')
    axs6[0].plot(acorr_run_50_bpm[0:31], "g-", label='Running (9.00 km/h)')
    axs6[0].plot([50]*30, 'k--', linewidth=1.5, label='Metronome Reference')

    axs6[0].set(
        title="Autocorrelation Breathing Rate Estimation (50 BPM)",
        ylabel="Breathing Rate (BPM)",
        xlabel="Time (s)"
    )
    axs6[0].set_ylim([48, 70])

    print("Acorr Walk 50BPM - Error: ", perc_error(acorr_walk_50_bpm[0:31], 50))
    print("Acorr Run 50BPM - Error: ", perc_error(acorr_run_50_bpm[0:31], 50))

    axs6[1].plot(fft_walk_50_bpm[20:51], "b-", label='Brisk Walk (5.25 km/h)')
    axs6[1].plot(fft_run_50_bpm[20:51], "g-", label='Running (9.00 km/h)')
    axs6[1].plot([50]*30, 'k--', linewidth=1.5, label='Metronome Reference')

    axs6[1].set(
        title="Doppler FFT Breathing Rate Estimation (50 BPM)",
        ylabel="Breathing Rate (BPM)",
        xlabel="Time (s)"
    )
    axs6[1].set_ylim([42, 57])

    print("FFT Walk 50BPM - Error: ", perc_error(fft_walk_50_bpm[20:51], 50))
    print("FFT Run 50BPM - Error: ", perc_error(fft_run_50_bpm[20:51], 50))

    axs6[0].legend()
    axs6[1].legend()
    axs6[0].legend(loc='lower left')

    fig6.set_figheight(4)
    fig6.set_figwidth(16)
    fig6.savefig("./Result Plots/Walking and Running at 50 BPM.png", dpi=200, bbox_inches='tight')

    # Plot Arm Movement Frequency (data from MATLAB)  -------------------------------------------------------------
    hz_data = [0.80808, 0.88725, 0.96579, 1.02433, 1.1168, 1.2, 1.321585, 1.34983, 1.3801, 1.43541, 1.46431] # Hz
    speed = [0.83, 1.03, 1.24, 1.46, 1.69, 1.88, 2.1, 2.37, 2.52, 2.7, 2.9] # m/s
    speed = np.multiply(speed, 3.6) # to km/h
    x_vals = np.linspace(0, 2.9*3.6)
    # ax + b
    a = 0.328498003233751
    b = 0.558818761181517
    y_vals = np.multiply(x_vals, a/3.6) # convert to km/h
    y_vals = np.add(y_vals, b)

    axs7.scatter(speed, hz_data, marker=".", color="b", label="Raw Data")
    axs7.plot(x_vals, y_vals, 'r--', label="Linear Fit")
    axs7.legend()

    axs7.set(
        title="Arm Movement Frequency vs Speed",
        ylabel="Frequency (Hz)",
        xlabel="Speed (km/h)"
    )

    font = { 'color': 'red' }
    axs7.text(4.5, 0.7, "f = 0.5588 + 0.0912*Speed", fontdict=font)

    fig5.set_figheight(1)
    fig5.set_figwidth(1)

    fig7.savefig("./Result Plots/Arm Movement.png", dpi=800, bbox_inches='tight')

    # plt.show()

# Speed and Radar Thread Functions ------------------------------------------------
def measure_speed():
    global current_speed
    while True:
        packet_length = int(recvall(speed_socket, HEADER_LENGTH).decode("utf-8"))
        current_speed = round(float(recvall(speed_socket, packet_length).decode("utf-8")), 2)

def measure_br():
    global radar_config, speed_tracker_active

    d_bin = int(DISTANCE//radar_config.range_resolution - 1) # Bin Corresponding to Distance

    fs = radar_config.frame_rate 

    max_rpm = 70 if speed_tracker_active else 30

    N = radar_config.num_samples_per_chirp
    M = fs * BUFFER_TIME
    zero_amount = ZERO_PAD - M # amount of zero padding to add to doppler fft

    chirp_buffer = np.zeros((N//2, M), dtype=complex)
    fft_buffer = np.zeros((N//2, ZERO_PAD//2), dtype=complex)

    br_data_fft = []
    br_data_acorr = []

    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(3)
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

        if save_data:
            # Save Matrix Data
            d_bin_buffer = chirp_buffer[d_bin]
            with open('buffer_{}.npy'.format(save_num), 'wb') as buffer_file:
                np.save(buffer_file, d_bin_buffer)

            # Save Speed
            with open('speed_{}.npy'.format(save_num), 'wb') as speed_file:
                np.save(speed_file, np.array([current_speed]))

            print("Last Saved: ", save_num)
            save_num = (save_num + 1)%10

        # DSP --------------------------------------------------------
        phase_array = get_phase_array(chirp_buffer[d_bin])

        # Apply Hanning window and FFT of slow time signal
        sig_hanning = np.multiply(phase_array, hann(M))
        sig_hanning = list(sig_hanning) + [0]*zero_amount
        fft_buffer = fft(sig_hanning)[0:ZERO_PAD//2]

        # Apply autocorrelation 
        acorr = dsp.auto_correlate(phase_array, fs)
        
        # Clear Graphs
        axs[0].cla()
        axs[1].cla()

        # Autocorrelation Prediction ------------------------------------------------
        acorr_prediction = 0
        auto_failed = False
        lin_r = np.abs(linregress(np.arange(1, M // 2), acorr)[2])
        if lin_r > MAX_LIN_R:
            auto_failed = True
    
        max_samples_per_breath = fs * (60 // max_rpm)
        peaks = None
        if max_samples_per_breath < 1:
            peaks = find_peaks(acorr, prominence=0.4)[0]
        else:
            peaks = find_peaks(acorr, distance=max_samples_per_breath, prominence=0.4)[0]
            
        if len(peaks) > 1:
            lag = peaks[1] - peaks[0]
            corr_val = acorr[peaks[1]]
            if corr_val < MIN_COR_VALUE:
                auto_failed = True
            else:
                acorr_prediction = 60*fs/lag
        else:
            auto_failed = True

        # Plotting Autocorrelation
        axs[0].plot(acorr) # TODO: Fix Axis
        if (len(peaks) > 0):
            axs[0].plot(peaks, acorr[peaks], 'xr')

        # FFT BR Prediction ------------------------------------------------
        fft_prediction = 0
        st_fft = np.abs(fft_buffer[0:ZERO_PAD//32])
        st_fft = np.divide(st_fft, max(st_fft))
        min_peak_distance = max_rpm*ZERO_PAD/(60*fs)

        fft_peaks = None
        if speed_tracker_active:
            fft_peaks = find_peaks(st_fft, prominence=0.05, height=0.7)[0] # WITH TREADMILL
        else:
            fft_peaks = find_peaks(st_fft, distance=min_peak_distance, prominence=0.15, height=0.5)[0] # NO TREADMILL

        if len(fft_peaks) > 0:
            fft_prediction = 1.15*60*fft_peaks[0]*fs/ZERO_PAD

        # Plotting FFT
        f_vals = np.linspace(0, fs/2, ZERO_PAD//2)[0:ZERO_PAD//32]
        axs[1].plot(f_vals, st_fft)
        if len(fft_peaks) > 0:
            axs[1].plot(np.multiply(fft_peaks, fs/ZERO_PAD), st_fft[fft_peaks], "xr")

        # Draw Plots
        plt.draw()
        plt.pause(0.000001)

        if auto_failed:
            acorr_prediction = 0

        # Capture and Save Breathing Array
        if save_data:    
            br_data_fft.append(fft_prediction) # Prediction is off by constant 15%
            br_data_acorr.append(acorr_prediction)

            with open('br_fft.npy', 'wb') as br_file:
                np.save(br_file, br_data_fft)

            with open('br_acorr.npy', 'wb') as br_file:
                np.save(br_file, br_data_acorr)

        print("Auto BR: {}, FFT BR: {}".format( round(acorr_prediction, 2), round(fft_prediction, 2)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hip', '--host_ip', default=HOST ,help='Raspberry Pi Host IP')
    parser.add_argument('-rp', '--radar_port', help='Radar Server Port')
    parser.add_argument('-sp', '--speed_port', default="0000", help='Speed Sensor Server Port')  # Assumes measurements for treadmill are taking palce
    args = parser.parse_args()
    speed_tracker_active = args.speed_port != "0000"
    
    if args.radar_port is None:
        print("Usage: python {command} -rp <radar-port> -sp <speed-tracker-port> -hip <rp-host-ip> -t (use this to activate treadmill)".format(command=sys.argv[0]))
        exit(0)

    # Setup Socket Connections and Start Threads
    RADAR_PORT = int(args.radar_port)
    radar_socket.connect((HOST, RADAR_PORT))
    print("Connected to Radar Socket: ", (HOST, RADAR_PORT))

    if speed_tracker_active:
        SPEED_PORT = int(args.speed_port)
        speed_socket.connect((HOST, SPEED_PORT))
        print("Connected to Speed Sensor Socket: ", (HOST, SPEED_PORT))

    if speed_tracker_active:
        speed_thread = Thread(target=measure_speed)
        speed_thread.start()

    configure_radar()
    measure_br()

    if speed_tracker_active:
        speed_thread.join()