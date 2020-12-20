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
HOST = "192.168.0.190"

# Global Variables ------------------------------------------------
current_speed = 0
radar_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
    exit(0)
signal.signal(signal.SIGINT, signal_handler)

def get_static_fft(samples=64):
    global radar_config

    N = radar_config.num_samples_per_chirp
    M = samples

    range_fft_matrix = np.zeros((M, N//2), dtype=complex)

    chirps_collected = 0
    while chirps_collected < M:
        # Fetch packet
        data_packet = fetch_json_packet(radar_socket)
        
        # Collect chirp
        chirp = data_packet["data"]
        chirp = detrend(chirp)

        # Store chirp data
        range_fft_matrix[chirps_collected] = fft(chirp)[0:N//2]

        chirps_collected += 1

        print("Chirp Collection Progress: {}%".format((100*chirps_collected)//M), end="\r")

    return range_fft_matrix.mean(0)

def radar_main_thread():
    global radar_config, dsp

    # Fetch config data first
    config_packet = fetch_json_packet(radar_socket)
    assert config_packet["packet_type"] == "config"

    # Set global variables based on config
    radar_config = RadarConfig(config_packet["data"])
    dsp = DigitalSignalProcessor(radar_config)

    static_fft = get_static_fft()

    dsp.range_fft_plot(static_fft, "static_capture")

    with open("static_clutter.csv", mode="w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(static_fft)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python {command} <radar-port> <speed-tracker-port> (optional)".format(command=sys.argv[0]))
        exit(0)

    # Setup Socket Connections and Start Threads
    RADAR_PORT = int(sys.argv[1])
    radar_socket.connect((HOST, RADAR_PORT))
    print("Connected to Radar Socket: ", (HOST, RADAR_PORT))

    radar_main_thread()