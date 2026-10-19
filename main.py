
import numpy as np
import matplotlib.pyplot as plt

from line_codes import (
    nrz_l,
    manchester,
    differential_manchester,
    ami
)


# Binary data to transmit
bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])

# Number of samples representing each bit
samples_per_bit = 100

# Generate encoded signals
nrz_signal = nrz_l(bits, samples_per_bit)

manchester_signal = manchester(
    bits, samples_per_bit
)

diff_manchester_signal = differential_manchester(
    bits, samples_per_bit
)

ami_signal = ami(bits, samples_per_bit)


# Time axis
time = np.arange(len(nrz_signal)) / samples_per_bit

# Create five plots
fig, axes = plt.subplots(
    5, 1,
    figsize=(12, 10),
    sharex=True
)


# Plot 1: Original binary data
axes[0].step(
    np.arange(len(bits) + 1),
    np.append(bits, bits[-1]),
    where="post"
)

axes[0].set_ylabel("Bit")
axes[0].set_title("Original Binary Data")
axes[0].set_yticks([0, 1])
axes[0].grid(True)


# Plot 2: NRZ-L
axes[1].plot(time, nrz_signal)

axes[1].set_ylabel("Voltage")
axes[1].set_title("NRZ-L")
axes[1].set_yticks([-1, 1])
axes[1].grid(True)


# Plot 3: Manchester
axes[2].plot(time, manchester_signal)

axes[2].set_ylabel("Voltage")
axes[2].set_title("Manchester")
axes[2].set_yticks([-1, 1])
axes[2].grid(True)


# Plot 4: Differential Manchester
axes[3].plot(time, diff_manchester_signal)

axes[3].set_ylabel("Voltage")
axes[3].set_title("Differential Manchester")
axes[3].set_yticks([-1, 1])
axes[3].grid(True)


# Plot 5: AMI
axes[4].plot(time, ami_signal)

axes[4].set_ylabel("Voltage")
axes[4].set_title("AMI")
axes[4].set_yticks([-1, 0, 1])
axes[4].set_xlabel("Time (bit periods)")
axes[4].grid(True)


# Improve layout and save the figure
plt.tight_layout()
plt.savefig("results_waveforms.png", dpi=300)
plt.show()
