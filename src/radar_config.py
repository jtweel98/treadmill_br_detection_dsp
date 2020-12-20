class RadarConfig:

    def __init__(self, config_dict):
        self.range_resolution = config_dict["range_resolution"]
        self.max_range = config_dict["max_range"]
        self.min_range = config_dict["min_range"]
        self.speed_resolution = config_dict["speed_resolution"]
        self.max_speed = config_dict["max_speed"]
        self.frame_rate = config_dict["frame_rate"]
        self.adc_sample_rate_hz = config_dict["adc_sample_rate_hz"]
        self.rx_antenna_number = config_dict["rx_antenna_number"]
        self.center_frequency = config_dict["center_frequency"]
        self.num_samples_per_chirp = config_dict["num_samples_per_chirp"]
        self.num_chirps_per_frame = config_dict["num_chirps_per_frame"]
        self.lower_frequency = config_dict["lower_frequency"]
        self.upper_frequency = config_dict["upper_frequency"]
        self.bandwidth = config_dict["bandwidth"]

    @property
    def chirp_length(self):
        return self.num_samples_per_chirp / self.adc_sample_rate_hz

    @property
    def chirp_slope(self):
        return self.bandwidth / self.chirp_length