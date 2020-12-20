from radar_config import RadarConfig
from scipy.fft import fft
from scipy import signal
import numpy as np
from copy import copy
from scipy.signal import detrend
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


C = 2.99792458e8

class DigitalSignalProcessor:
    def __init__(self, radar_config):
        self.radar_config = radar_config
    
    def hp_filter_butterworth(self, sig, cutoff, fs, order=5):
        sos = signal.butter(order, cutoff, 'hp', fs=fs, output='sos')
        return signal.sosfilt(sos, sig)

    def lp_filter_butterworth(self, sig, cutoff, fs, order=5):
        sos = signal.butter(order, cutoff, 'lp', fs=fs, output='sos')
        return signal.sosfilt(sos, sig)
    
    def bp_filter_butterworth(self, sig, low_cutoff, high_cutoff, fs, order=5):
        sos = signal.butter(order, [low_cutoff, high_cutoff], 'bandpass', fs=fs, output='sos')
        return signal.sosfilt(sos, sig)

    def range_fft_plot(self, sig, title):
        range_fft_fig = plt.figure()
        range_fft_ax = range_fft_fig.add_subplot(111)
        range_fft_x_values = np.linspace(
            self.radar_config.range_resolution*39.3701, self.radar_config.max_range*39.3701, len(sig)
        )
        range_fft_ax.set(ylabel="Itensity", xlabel="Distance", ylim=[0, 1])
        range_fft_ax.plot(range_fft_x_values, np.abs(sig), lw=3)
        plt.savefig(title, dpi=80, bbox_inches='tight')

    def range_fft_gif(self, fft_buffer, title_afix="Default"):
        range_fft_fig = plt.figure()
        range_fft_ax = range_fft_fig.add_subplot(111)
        range_fft_x_values = np.linspace(
            self.radar_config.range_resolution*39.3701, self.radar_config.max_range*39.3701, len(fft_buffer[0])
        )
        range_fft_ax.set(ylabel="Itensity", xlabel="Distance")
        line, = range_fft_ax.plot(range_fft_x_values, [0]*len(range_fft_x_values), lw=3)
        def init():
            line.set_data(range_fft_x_values, [0]*len(range_fft_x_values))
            return line,
        def animate(i):
            data = np.abs(fft_buffer[i])
            line.set_ydata(data)
            range_fft_ax.set(
                title="Time: {} s".format(round(i/self.radar_config.frame_rate, 2)),
                ylim=[0, max([1, max(data)])]
            )
            return line,
        anim = FuncAnimation(
            range_fft_fig, 
            animate,
            init_func=init,
            frames=len(fft_buffer[:, 0]),
            interval=len(fft_buffer[:, 0])//self.radar_config.frame_rate,
            blit=True
        )
        anim.save("Range FFT {}.gif".format(title_afix))
    
    def range_bin_plot(self, fft_buffer, centre_distance, delta=2, title_afix="Default"):
        rr = self.radar_config.range_resolution
        fr = self.radar_config.frame_rate
        total_plots = 1 + 2*delta
        fig, axs = plt.subplots(total_plots, figsize=(13, 13))
        fig.tight_layout(pad=3)
        centre_bin = int(centre_distance//rr) - 1
        t_vals = np.linspace(0, len(fft_buffer[:, 0])//fr, len(fft_buffer[:, 0]))
        for i in range(total_plots):
            curr_bin = centre_bin - delta + i
            data = fft_buffer[:, curr_bin]
            data = np.abs(data)
            data = detrend(data)
            axs[i].set_title("Distance: {}, Bin: {}".format(round((curr_bin + 1)*rr, 2), curr_bin))
            axs[i].plot(t_vals, data)

            plt.savefig("Bin Time Series " + title_afix, dpi=80, bbox_inches='tight')

    def doppler_fft_bin_plot(self, doppler_fft_buffer, centre_distance, delta=2, title_afix="Default"):
        rr = self.radar_config.range_resolution
        fr = self.radar_config.frame_rate
        half_M = len(doppler_fft_buffer[0])
        total_plots = 1 + 2*delta
        fig, axs = plt.subplots(total_plots, figsize=(13, 13))
        fig.tight_layout(pad=3)
        centre_bin = int(centre_distance//rr) - 1
        f_vals = np.linspace(0, fr//2, half_M)
        for i in range(total_plots):
            curr_bin = centre_bin - delta + i
            doppler_data = doppler_fft_buffer[curr_bin]
            axs[i].set_title("Distance: {}, Bin: {}".format(round((curr_bin + 1)*rr, 2), curr_bin))
            axs[i].plot(f_vals, np.abs(doppler_data))

            plt.savefig("Bin Doppler FFT " + title_afix, dpi=80, bbox_inches='tight')

    def auto_correlate(self, signal, fs, offline_plot=False):
        N = signal.size
        variance = signal.var()

        # Return if no variance
        if variance == 0:
            return [0] * N
        
        # Perform autocorrelation
        corr_result = np.correlate(signal, signal, mode="same")

        if offline_plot:
            acorr = np.divide(corr_result, max(corr_result))
            acorr = acorr[N//2:]
            acorr = detrend(acorr, type="linear")
            return acorr

        # Center autocorrelation with lag 1 at index 0
        acorr = corr_result[N//2+1:] / (variance * np.arange(N-1, N//2, -1))

        # Linear Detrend Signal
        acorr = detrend(acorr, type="linear")

        # Filter coefficient spectrum
        # acorr = self.lp_filter_butterworth(sig=acorr, cutoff=3, fs=fs)

        return acorr

    def decouple_dc(self, sig):
        sig -= np.mean(sig)
        return sig

    def ss_fft(self, sig, zero_pad=0):
        # single side fft
        sig_cpy = list(copy(sig))

        if zero_pad > 0:
            sig_cpy += (zero_pad - len(sig))*[0]

        fs = self.radar_config.adc_sample_rate_hz
        N = len(sig_cpy)
        freq_spec = fft(sig_cpy)
        ss_freq_spec = freq_spec[0:N // 2]
        x_f = np.linspace(0, fs / 2, N // 2)

        return ss_freq_spec, x_f, N // 2

    def if_to_d(self, if_sig):
        S = self.radar_config.chirp_slope
        return C * if_sig / (2.0 * S)
    
    def d_to_if(self, dis):
        S = self.radar_config.chirp_slope
        return S * 2.0 * dis / C
    
    def filter_min_distance(self, sig):
        fc = self.d_to_if(self.radar_config.min_range)
        return self.hp_filter_butterworth(sig, fc)
