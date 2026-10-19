import numpy as np

def decode_nrz_l(received, samples_per_bit=100):
    received = np.asarray(received)
    bits = []
    for start in range(0, len(received), samples_per_bit):
        chunk = received[start:start + samples_per_bit]

        if len(chunk) < samples_per_bit:
            break

        bits.append(1 if np.mean(chunk) > 0 else 0)

    return np.array(bits, dtype=int)


def decode_manchester(received, samples_per_bit=100):
    received = np.asarray(received)
    if samples_per_bit <= 0 or samples_per_bit % 2 != 0:
        raise ValueError("samples_per_bit must be a positive even number.")

    bits = []
    half_bit = samples_per_bit // 2

    for start in range(0, len(received), samples_per_bit):
        chunk = received[start:start + samples_per_bit]

        if len(chunk) < samples_per_bit:
            break

        first_half = np.mean(chunk[:half_bit])
        second_half = np.mean(chunk[half_bit:])

        bits.append(1 if first_half > second_half else 0)

    return np.array(bits, dtype=int)


def decode_differential_manchester(received, samples_per_bit=100):
    received = np.asarray(received)

    if samples_per_bit <= 0 or samples_per_bit % 2 != 0:
        raise ValueError(
            "samples_per_bit must be a positive even number."
        )

    bits = []
    half_bit = samples_per_bit // 2

    previous_end_level = 1

    for start in range(0, len(received), samples_per_bit):
        chunk = received[start:start + samples_per_bit]

        if len(chunk) < samples_per_bit:
            break

        first_half = np.mean(chunk[:half_bit])
        second_half = np.mean(chunk[half_bit:])

        # Estimate the received level in each half-bit
        first_level = 1 if first_half >= 0 else -1
        second_level = 1 if second_half >= 0 else -1

        # A transition at the start means bit 0.
        # No transition at the start means bit 1.
        bit = 1 if first_level == previous_end_level else 0

        bits.append(bit)

        # The second half's level becomes the reference
        # for the next bit.
        previous_end_level = second_level

    return np.array(bits, dtype=int)


def decode_ami(received, samples_per_bit=100):
    received = np.asarray(received)

    if samples_per_bit <= 0:
        raise ValueError("samples_per_bit must be positive.")

    bits = []

    for start in range(0, len(received), samples_per_bit):
        chunk = received[start:start + samples_per_bit]

        if len(chunk) < samples_per_bit:
            break

        mean_voltage = np.mean(chunk)

        # AMI: binary 0 is represented by zero voltage;
        # binary 1 is represented by either positive or negative pulses.
        bits.append(1 if abs(mean_voltage) > 0.5 else 0)

    return np.array(bits, dtype=int)


def calculate_ber(original_bits, decoded_bits):
    original_bits = np.asarray(original_bits)
    decoded_bits = np.asarray(decoded_bits)
    
    if len(original_bits) != len(decoded_bits):
        raise ValueError(
            "Original and decoded bit sequences must have equal lengths."
        )

    if len(original_bits) == 0:
        raise ValueError("Bit sequence cannot be empty.")

    errors = np.sum(original_bits != decoded_bits)

    return errors / len(original_bits)