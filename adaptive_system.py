import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "models/classifier.pkl"

NUMBER_OF_BITS = 1000
SAMPLES_PER_BIT = 100

ENCODERS = {
    "NRZ-L": nrz_l,
    "Manchester": manchester,
    "Differential Manchester": differential_manchester,
    "AMI": ami
}

DECODERS = {
    "NRZ-L": decode_nrz_l,
    "Manchester": decode_manchester,
    "Differential Manchester": decode_differential_manchester,
    "AMI": decode_ami
}


# --------------------------------------------------
# LOAD TRAINED ML MODEL
# --------------------------------------------------

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


# --------------------------------------------------
# GET CHANNEL CONDITIONS FROM USER
# --------------------------------------------------

def get_channel_conditions():

    print("\nSelect channel type:")
    print("1. AWGN")
    print("2. Rayleigh")
    print("3. Rician")
    print("4. Whole Signal Polarity Inversion")
    print("5. Per-Bit Pulse Polarity Disturbance")

    while True:
        try:
            choice = int(input("Enter choice (1-5): "))

            if choice in range(1, 6):
                break

            print("Choose a number from 1 to 5.")

        except ValueError:
            print("Enter a valid integer.")

    channel_names = {
        1: "AWGN",
        2: "Rayleigh",
        3: "Rician",
        4: "Whole Signal Polarity Inversion",
        5: "Per-Bit Pulse Polarity Disturbance"
    }

    channel_type = channel_names[choice]

    while True:
        try:
            snr_db = float(
                input("Enter SNR in dB (-5 to 20): ")
            )

            if -5 <= snr_db <= 20:
                break

            print("Enter an SNR between -5 and 20 dB.")

        except ValueError:
            print("Enter a valid number.")

    # Obtain K-factor only for Rician fading.
    if channel_type == "Rician":

        while True:
            try:
                k_factor = float(
                    input("Enter K-factor (1, 3, or 10): ")
                )

                if k_factor in [1.0, 3.0, 10.0]:
                    break

                print("Choose 1, 3, or 10.")

            except ValueError:
                print("Enter a valid number.")

    else:
        k_factor = 0.0

    if channel_type in [
        "Whole Signal Polarity Inversion",
        "Per-Bit Pulse Polarity Disturbance"
    ]:
        impairment = channel_type

    else:
        impairment = "none"

    return snr_db, channel_type, k_factor, impairment


# --------------------------------------------------
# ML MODEL PREDICTION
# --------------------------------------------------

def select_scheme(
    model,
    snr_db,
    channel_type,
    k_factor,
    impairment
):

    inputs = pd.DataFrame([{
        "snr_db": snr_db,
        "channel_type": channel_type,
        "rician_k_factor": k_factor,
        "impairment": impairment
    }])

    prediction = model.predict(inputs)[0]

    return prediction


# --------------------------------------------------
# TRANSMIT SIGNAL THROUGH SELECTED CHANNEL
# --------------------------------------------------

def transmit_signal(
    bits,
    scheme,
    channel_type,
    snr_db,
    k_factor
):

    encoder = ENCODERS[scheme]
    decoder = DECODERS[scheme]

    # Encode the original information bits.
    encoded_signal = encoder(
        bits,
        SAMPLES_PER_BIT
    )

    # Apply the selected channel.
    if channel_type == "AWGN":

        received_signal = add_awgn(
            encoded_signal,
            snr_db,
            SAMPLES_PER_BIT
        )

    elif channel_type == "Rayleigh":

        received_signal = add_rayleigh(
            encoded_signal,
            snr_db,
            SAMPLES_PER_BIT
        )

    elif channel_type == "Rician":

        received_signal = add_rician(
            encoded_signal,
            snr_db,
            SAMPLES_PER_BIT,
            k_factor=k_factor
        )

    elif channel_type == "Whole Signal Polarity Inversion":

        # Reverse the polarity of the entire encoded signal.
        received_signal = -encoded_signal

        received_signal = add_awgn(
            received_signal,
            snr_db,
            SAMPLES_PER_BIT
        )

    elif channel_type == "Per-Bit Pulse Polarity Disturbance":

        # Randomly reverse the polarity of individual bit intervals.
        received_signal = encoded_signal.copy()

        for start in range(
            0,
            len(received_signal),
            SAMPLES_PER_BIT
        ):

            end = start + SAMPLES_PER_BIT

            if np.random.random() < 0.5:
                received_signal[start:end] *= -1

        received_signal = add_awgn(
            received_signal,
            snr_db,
            SAMPLES_PER_BIT
        )

    else:
        raise ValueError("Unsupported channel type.")

    # Decode the received signal.
    decoded_bits = decoder(
        received_signal,
        SAMPLES_PER_BIT
    )

    # Calculate Bit Error Rate.
    ber = calculate_ber(
        bits,
        decoded_bits
    )

    return decoded_bits, ber


# --------------------------------------------------
# COMPARE ALL FOUR LINE CODING SCHEMES
# --------------------------------------------------

def compare_all_schemes(
    bits,
    channel_type,
    snr_db,
    k_factor
):

    results = {}

    for scheme in ENCODERS:

        # Reset the random seed for reproducible comparisons.
        np.random.seed(123)

        _, ber = transmit_signal(
            bits,
            scheme,
            channel_type,
            snr_db,
            k_factor
        )

        results[scheme] = ber

    return results


# --------------------------------------------------
# PLOT AND SAVE BER COMPARISON GRAPH
# --------------------------------------------------

def plot_ber_comparison(
    results,
    selected_scheme,
    best_scheme,
    channel_type,
    snr_db
):

    schemes = list(results.keys())
    bers = list(results.values())

    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        schemes,
        bers
    )

    # Highlight the ML-selected scheme.
    selected_index = schemes.index(selected_scheme)
    bars[selected_index].set_color("orange")

    # Outline the lowest-BER scheme.
    best_index = schemes.index(best_scheme)
    bars[best_index].set_edgecolor("black")
    bars[best_index].set_linewidth(2)

    plt.xlabel(
        "Line Coding Scheme",
        fontsize=12
    )

    plt.ylabel(
        "Bit Error Rate (BER)",
        fontsize=12
    )

    plt.title(
        f"BER Comparison\nChannel: {channel_type}, SNR: {snr_db} dB",
        fontsize=14
    )

    # Leave space above bars for their BER labels.
    maximum_ber = max(bers)

    if maximum_ber > 0:
        plt.ylim(0, maximum_ber * 1.20)
    else:
        plt.ylim(0, 0.01)

    plt.xticks(rotation=12)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    # Display the numerical BER above each bar.
    for bar, ber in zip(bars, bers):

        label_height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            label_height,
            f"{ber:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    # Save the graph in the current working directory.
    filename = "results_ber_comparison.png"

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"\nBER graph saved as: {filename}")
    print("Orange bar = ML-selected scheme")
    print("Black outline = lowest-BER scheme")

    # Display the graph.
    plt.show()

    plt.close()


# --------------------------------------------------
# RUN ONE SIMULATION
# --------------------------------------------------

def main():

    print("\n" + "=" * 65)
    print("          SMART LINE CODING SELECTION SYSTEM")
    print("=" * 65)

    # Load the trained model.
    try:
        model = load_model()

    except FileNotFoundError as error:
        print(error)
        return

    # Step 1: Obtain channel conditions.
    (
        snr_db,
        channel_type,
        k_factor,
        impairment
    ) = get_channel_conditions()

    # Step 2: Predict the line coding scheme.
    selected_scheme = select_scheme(
        model,
        snr_db,
        channel_type,
        k_factor,
        impairment
    )

    # Step 3: Generate information bits.
    rng = np.random.default_rng(42)

    bits = rng.integers(
        0,
        2,
        size=NUMBER_OF_BITS
    )

    print("\n" + "=" * 65)
    print("                    TRANSMISSION")
    print("=" * 65)

    print(f"{'Channel type:':<30} {channel_type}")
    print(f"{'SNR:':<30} {snr_db} dB")

    if channel_type == "Rician":
        print(f"{'Rician K-factor:':<30} {k_factor}")

    print(f"{'Number of transmitted bits:':<30} {NUMBER_OF_BITS}")
    print(f"{'ML-predicted scheme:':<30} {selected_scheme}")

    # Step 4: Evaluate every coding scheme.
    print("\nEvaluating all four coding schemes...")

    results = compare_all_schemes(
        bits,
        channel_type,
        snr_db,
        k_factor
    )

    # Step 5: Display BER results.
    print("\n" + "=" * 65)
    print("                     BER COMPARISON")
    print("=" * 65)

    for scheme, ber in results.items():

        print(
            f"{scheme:<30} BER = {ber:.6f}"
        )

    # Step 6: Find the scheme with the lowest measured BER.
    best_scheme = min(
        results,
        key=results.get
    )

    best_ber = results[best_scheme]

    adaptive_ber = results[selected_scheme]

    # Step 7: Display the final results.
    print("\n" + "=" * 65)
    print("                      FINAL RESULTS")
    print("=" * 65)

    print(
        f"{'ML-predicted scheme:':<30} {selected_scheme}"
    )

    print(
        f"{'ML-selected scheme BER:':<30} {adaptive_ber:.6f}"
    )

    print(
        f"{'Best-performing scheme:':<30} {best_scheme}"
    )

    print(
        f"{'Lowest measured BER:':<30} {best_ber:.6f}"
    )

    if selected_scheme == best_scheme:

        print(
            "\nThe ML model selected the lowest-BER scheme "
            "in this trial."
        )

    else:

        print(
            "\nThe ML model did not select the lowest-BER "
            "scheme in this trial."
        )

    print("=" * 65)

    # Step 8: Plot and save the BER comparison graph.
    plot_ber_comparison(
        results,
        selected_scheme,
        best_scheme,
        channel_type,
        snr_db
    )


# --------------------------------------------------
# ASK WHETHER THE USER WANTS ANOTHER SIMULATION
# --------------------------------------------------

if __name__ == "__main__":

    while True:

        main()

        while True:

            choice = input(
                "\nDo you want to test another channel condition? "
                "(yes/no): "
            ).strip().lower()

            if choice in ("yes", "y"):
                print("\nStarting another simulation...\n")
                break

            elif choice in ("no", "n"):

                print(
                    "\nExiting Smart Line Coding Selection System."
                )

                raise SystemExit(0)

            else:

                print(
                    "Invalid input. Please enter yes or no."
                )