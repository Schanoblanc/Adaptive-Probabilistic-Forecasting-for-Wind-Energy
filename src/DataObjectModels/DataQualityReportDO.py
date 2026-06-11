import numpy as np

class DQSum:
    def __init__(self, windfarm, year, spike_threshold_config, epsilon_config, max_spike_pcent, 
                nan_count, nan_pcent, zero_count, zero_pcent, spike_count,spike_pcent, epsilon_count,epsilon_pcent,
                cleanmax, clean_epsilon_max_count,clean_epsilon_max_pcent):
        self.Windfarm = windfarm
        self.Year = year
        self.SpikeThresholdConfig = spike_threshold_config
        self.MaxSpikePcentConfig = max_spike_pcent
        self.EpsilonConfig = epsilon_config
        self.NanCount = nan_count
        self.NanPcent = nan_pcent
        self.ZeroCount = zero_count
        self.ZeroPcent = zero_pcent
        self.SpikeCount = spike_count
        self.SpikeCPcent = spike_pcent
        self.EplisonCount = epsilon_count
        self.EplisonPcent = epsilon_pcent
        self.CleanMax = cleanmax
        self.CleanEpsilonMaxCount = clean_epsilon_max_count
        self.CleanEpsilonMaxPcent = clean_epsilon_max_pcent

    def ToDictionary(self):
        return {
            "Windfarm": self.Windfarm,
            "Year": self.Year,
            "SpikeThresholdConfig": self.SpikeThresholdConfig,
            "EpsilonConfig": self.EpsilonConfig,
            "MaxSpikePcentConfig": self.MaxSpikePcentConfig,
            "NanCount": self.NanCount,
            "NanPcent": self.NanPcent,
            "ZeroCount": self.ZeroCount,
            "ZeroPcent": self.ZeroPcent,
            "SpikeCount": self.SpikeCount,
            "SpikeCPcent": self.SpikeCPcent,
            "EplisonCount": self.EplisonCount,
            "EplisonPcent": self.EplisonPcent,
            "CleanMax": self.CleanMax,
            "CleanEpsilonMaxCount": self.CleanEpsilonMaxCount,
            "CleanEpsilonMaxPcent": self.CleanEpsilonMaxPcent
        }