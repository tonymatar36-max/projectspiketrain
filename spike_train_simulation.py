# -----------------------------
# Spike Train Simulation Project
# -----------------------------
# Author: Tony M (for portfolio)
# Purpose: Simulate spike trains, compute firing rates, population stats, detect bursts
# -----------------------------

import numpy as np

# -----------------------------
# Simulation parameters
# -----------------------------
np.random.seed(0)  # for reproducibility

n_neurons = 40        # number of neurons
n_timepoints = 800    # number of time bins (ms)
firing_prob = 0.04    # probability of firing per bin
delta_t = 0.001       # duration of one time bin in seconds (1 ms)

# -----------------------------
# Generate spike matrix
# -----------------------------
# Binary matrix: 1 = spike, 0 = no spike
spikes = (np.random.rand(n_neurons, n_timepoints) < firing_prob).astype(int)
print("Spike matrix shape:", spikes.shape)  # (neurons, timepoints)

# -----------------------------
# Per-neuron statistics
# -----------------------------
spike_count = np.sum(spikes, axis=1)
spike_mean = np.mean(spikes, axis=1)
firing_rate = spike_mean / delta_t  # in Hz
spike_variance = np.var(spikes, axis=1)

print("\n=== Per-neuron statistics ===")
print("Spike counts per neuron:", spike_count)
print("Mean spikes per neuron:", spike_mean)
print("Firing rate per neuron (Hz):", firing_rate)
print("Variance per neuron:", spike_variance)

# -----------------------------
# Population-level analysis
# -----------------------------
population_activity = np.sum(spikes, axis=0)  # sum across neurons per time bin
burst_threshold = 5                           # define a population burst
bursts = np.where(population_activity > burst_threshold)[0]

print("\n=== Population statistics ===")
print("Population activity per time bin:", population_activity)
print("Time bins with population bursts:", bursts)

# -----------------------------
# Most active neuron using np.where
# -----------------------------
# Handles ties automatically
most_active_neurons = np.where(spike_count == np.max(spike_count))[0]
most_active_neuron = most_active_neurons[0]  # first if tie

print("\nMost active neuron(s) index:", most_active_neurons)
print("First most active neuron index:", most_active_neuron)

# -----------------------------
# Optional extras
# -----------------------------
# Average population firing rate
avg_population_rate = np.mean(population_activity) / delta_t
print("\nAverage population firing rate (Hz):", avg_population_rate)

# Variance/Mean ratio to check Poisson-like firing
poisson_like_ratio = spike_variance / spike_mean
print("Variance/Mean ratio per neuron (Poisson check):", poisson_like_ratio)

# -----------------------------
# End of Project
# -----------------------------
print("\n Simulation & analysis complete! Ready to add to LinkedIn portfolio.")
