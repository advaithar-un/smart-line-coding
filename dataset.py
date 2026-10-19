import os
import numpy as np
import pandas as pd

from line_codes import (
    nrz_l,
    manchester,
    differential_manchester,
    ami
)

from channel import (
    add_awgn,
    add_rayleigh,
    add_rician
)

from receiver import (
    decode_nrz_l,
    decode_manchester,
    decode_differential_manchester,
    decode_ami,
    calculate_ber
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUMBER_OF_BITS = 10000
SAMPLES_PER_BIT = 100

SNR_VALUES = list(range(-5, 21, 5))
NUMBER_OF_TRIALS = 10

RANDOM_SEED = 42
RICIAN_K_FACTORS = [1.0, 3.0, 10.0]


# --------------------------------------------------
# Line coding schemes
# --------------------------------------------------

SCHEMES = {
    "NRZ-L": (nrz_l, decode_nrz_l),
    "Manchester": (manchester, decode_manchester),
    "Differential Manchester": (
        differential_manchester,
        decode_differential_manchester
    ),
    "AMI": (ami, decode_ami)
}


# --------------------------------------------------
# Channel configurations
# --------------------------------------------------

CHANNELS = [
    ("AWGN", add_awgn, None, "none"),
    ("Rayleigh", add_rayleigh, None, "none")
]

for k_factor in RICIAN_K_FACTORS:
    CHANNELS.append(
        ("Rician", add_rician, k_factor, "none")
    )

# Special channel conditions.
# Both use AWGN before applying the additional impairment.

CHANNELS.append(
    (
        "Whole Signal Polarity Inversion",
        add_awgn,
        None,
        "whole_signal_inversion"
    )
)

CHANNELS.append(
    (
        "Per-Bit Pulse Polarity Disturbance",
        add_awgn,
        None,
        "per_bit_polarity_disturbance"
    )
)


# --------------------------------------------------
# Channel impairments
# --------------------------------------------------

def apply_impairment(signal, impairment):
    signal = np.asarray(signal, dtype=float)

    if impairment == "none":
        return signal

    if impairment == "whole_signal_inversion":
        # Reverse the polarity of the entire received waveform.
        return -signal

    if impairment == "per_bit_polarity_disturbance":
        # Independently reverse the polarity of each bit interval.
        # This tests whether the decoder depends on pulse polarity.
        signal = signal.copy()

        for start in range(
            0,
            len(signal),
            SAMPLES_PER_BIT
        ):
            end = start + SAMPLES_PER_BIT

            if np.random.random() < 0.5:
                signal[start:end] *= -1

        return signal

    raise ValueError(f"Unknown impairment: {impairment}")


# --------------------------------------------------
# BER simulation
# --------------------------------------------------

def simulate_ber(
    bits,
    encoder,
    decoder,
    channel_function,
    snr_db,
    k_factor=None,
    impairment="none"
):
    encoded_signal = encoder(
        bits,
        SAMPLES_PER_BIT
    )

    if k_factor is None:
        received_signal = channel_function(
            encoded_signal,
            snr_db,
            SAMPLES_PER_BIT
        )
    else:
        received_signal = channel_function(
            encoded_signal,
            snr_db,
            SAMPLES_PER_BIT,
            k_factor=k_factor
        )

    received_signal = apply_impairment(
        received_signal,
        impairment
    )

    decoded_bits = decoder(
        received_signal,
        SAMPLES_PER_BIT
    )

    return calculate_ber(bits, decoded_bits)


# --------------------------------------------------
# Dataset generation
# --------------------------------------------------

def generate_dataset():

    rng = np.random.default_rng(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    rows = []

    total_tests = (
        len(CHANNELS)
        * len(SNR_VALUES)
        * NUMBER_OF_TRIALS
    )

    completed_tests = 0

    print("Starting dataset generation...")
    print(f"Channel configurations: {len(CHANNELS)}")
    print(f"SNR values: {SNR_VALUES}")
    print(f"Trials per configuration: {NUMBER_OF_TRIALS}")
    print(f"Bits per trial: {NUMBER_OF_BITS}")
    print(f"Total channel/trial combinations: {total_tests}")
    print()

    for (
        channel_name,
        channel_function,
        k_factor,
        impairment
    ) in CHANNELS:

        for snr in SNR_VALUES:

            for trial in range(NUMBER_OF_TRIALS):

                bits = rng.integers(
                    0,
                    2,
                    size=NUMBER_OF_BITS
                )

                ber_results = {}

                for scheme_name, functions in SCHEMES.items():

                    encoder, decoder = functions

                    ber = simulate_ber(
                        bits=bits,
                        encoder=encoder,
                        decoder=decoder,
                        channel_function=channel_function,
                        snr_db=snr,
                        k_factor=k_factor,
                        impairment=impairment
                    )

                    ber_results[scheme_name] = ber

                # Select the scheme with the lowest measured BER.
                best_scheme = min(
                    ber_results,
                    key=ber_results.get
                )

                rows.append({
                    "snr_db": snr,
                    "channel_type": channel_name,
                    "rician_k_factor": (
                        k_factor if k_factor is not None else 0.0
                    ),
                    "impairment": impairment,
                    "trial": trial + 1,
                    "ber_nrz_l": ber_results["NRZ-L"],
                    "ber_manchester": ber_results["Manchester"],
                    "ber_differential_manchester":
                        ber_results["Differential Manchester"],
                    "ber_ami": ber_results["AMI"],
                    "best_scheme": best_scheme
                })

                completed_tests += 1

                print(
                    f"Progress: {completed_tests}/{total_tests} | "
                    f"Channel: {channel_name} | "
                    f"SNR: {snr} dB | "
                    f"Trial: {trial + 1} | "
                    f"Best: {best_scheme}"
                )

    # Save dataset.
    os.makedirs("data", exist_ok=True)

    df = pd.DataFrame(rows)

    output_path = os.path.join(
        "data",
        "channel_dataset.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nDataset generation completed.")
    print(f"Saved to: {output_path}")
    print(f"Total rows: {len(df)}")

    print("\nChannel distribution:")
    print(df["channel_type"].value_counts())

    print("\nOverall best-scheme distribution:")
    print(df["best_scheme"].value_counts())

    print("\nBest scheme by channel condition:")
    print(
        df.groupby(
            ["channel_type", "best_scheme"]
        ).size().unstack(fill_value=0)
    )

    print("\nAverage BER by channel condition:")
    print(
        df.groupby("channel_type")[
            [
                "ber_nrz_l",
                "ber_manchester",
                "ber_differential_manchester",
                "ber_ami"
            ]
        ].mean().round(5)
    )


if __name__ == "__main__":
    generate_dataset()