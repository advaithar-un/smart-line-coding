
import numpy as np


def nrz_l(bits, samples_per_bit=100):
    """
    NRZ-L:
    1 -> +1
    0 -> -1
    """
    signal = []

    for bit in bits:
        voltage = 1 if bit == 1 else -1
        signal.extend([voltage] * samples_per_bit)

    return np.array(signal)


def manchester(bits, samples_per_bit=100):
    """
    Manchester:
    1 -> +1 followed by -1
    0 -> -1 followed by +1
    Each bit has a middle transition.
    """
    signal = []
    half_bit = samples_per_bit // 2

    for bit in bits:
        if bit == 1:
            signal.extend([1] * half_bit)
            signal.extend([-1] * half_bit)
        else:
            signal.extend([-1] * half_bit)
            signal.extend([1] * half_bit)

    return np.array(signal)


def differential_manchester(bits, samples_per_bit=100):
    """
    Differential Manchester:
    0 -> transition at the beginning
    1 -> no transition at the beginning
    Every bit has a middle transition.
    """
    signal = []
    half_bit = samples_per_bit // 2
    current_level = 1

    for bit in bits:
        # Transition at the start for bit 0
        if bit == 0:
            current_level *= -1

        # First half of the bit
        signal.extend([current_level] * half_bit)

        # Mandatory middle transition
        current_level *= -1

        # Second half of the bit
        signal.extend([current_level] * half_bit)

    return np.array(signal)


def ami(bits, samples_per_bit=100):
    """
    AMI:
    0 -> zero voltage
    1 -> alternating positive and negative pulses
    """
    signal = []
    last_voltage = -1

    for bit in bits:
        if bit == 1:
            last_voltage *= -1
            voltage = last_voltage
        else:
            voltage = 0

        signal.extend([voltage] * samples_per_bit)

    return np.array(signal)
