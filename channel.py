
import numpy as np


def _add_noise(signal, snr_db, samples_per_bit):
    signal = np.asarray(signal, dtype=float)

    if samples_per_bit <= 0:
        raise ValueError("samples_per_bit must be positive.")

    signal_power = np.mean(signal ** 2)

    if signal_power <= 0:
        raise ValueError("Signal power must be positive.")

    # Interpret snr_db as Eb/N0.
    # For an oversampled signal, noise variance per sample
    # must account for the samples per bit.
    eb_n0 = 10 ** (snr_db / 10)

    noise_power = signal_power * samples_per_bit / (2 * eb_n0)
    noise_std = np.sqrt(noise_power)

    noise = np.random.normal(
        0.0,
        noise_std,
        size=signal.shape
    )

    return signal + noise


def add_awgn(signal, snr_db, samples_per_bit=100):
    return _add_noise(signal, snr_db, samples_per_bit)


def add_rayleigh(signal, snr_db, samples_per_bit=100):
    signal = np.asarray(signal, dtype=float)

    fading = np.random.rayleigh(
        scale=1 / np.sqrt(2),
        size=signal.shape
    )

    faded_signal = signal * fading

    return _add_noise(
        faded_signal,
        snr_db,
        samples_per_bit
    )


def add_rician(
    signal,
    snr_db,
    samples_per_bit=100,
    k_factor=3.0
):
    signal = np.asarray(signal, dtype=float)

    if k_factor < 0:
        raise ValueError("k_factor must be non-negative.")

    los = np.sqrt(k_factor / (k_factor + 1))
    scatter = np.sqrt(1 / (k_factor + 1))

    random_component = np.random.normal(
        0.0,
        1 / np.sqrt(2),
        size=signal.shape
    )

    fading = los + scatter * random_component
    faded_signal = signal * fading

    return _add_noise(
        faded_signal,
        snr_db,
        samples_per_bit
    )
