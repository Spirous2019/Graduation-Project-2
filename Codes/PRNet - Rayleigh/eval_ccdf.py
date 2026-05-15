"""Evaluate the paper-compatible CCDF for the Rayleigh-trained PRNet model.

The original PRNet paper's red CCDF curve is reproduced by the metric used in
the authors' released code: 10*log10(max(|x|)) after RMS normalizing the PRNet
waveform. That is a peak-amplitude proxy rather than the standard power-PAPR
definition, but it is the metric needed to match the paper's Fig. 3.

This script intentionally plots only one PRNet curve: a solid red PRNet curve
using the paper-compatible metric. The original OFDM baseline is retained as a
black dashed curve for comparison.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ofdm_helpers import (
    compute_ccdf,
    compute_papr_amplitude_proxy_db,
    compute_papr_db,
    generate_qpsk_symbols,
    ofdm_ifft,
    symbols_to_real,
)
from prnet_model import PRNet


N = 64
CCDF_OVERSAMPLING = 1
NUM_SYMBOLS = 25_000
LAMBDA = 0.01
PAPR_THRESHOLDS = np.linspace(0, 14, 300)
RANDOM_SEED = 2026

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def percentile_at_ccdf(papr_values: np.ndarray, ccdf: float = 1e-3) -> float:
    """Return the PAPR threshold whose empirical CCDF is approximately ccdf."""
    return float(np.quantile(papr_values, 1.0 - ccdf))


def load_model(oversampling: int) -> PRNet:
    model = PRNet(
        N=N,
        L=oversampling,
        num_blocks=5,
        hidden_dim=2048,
        lambda_=LAMBDA,
        eta_db=-15.0,
    )
    model(tf.zeros((1, 2 * N)), training=False)

    wpath = os.path.join(MODEL_DIR, f"prnet_lambda_{LAMBDA}.weights.h5")
    if os.path.exists(wpath):
        model.load_weights(wpath)
        print(f"[ok] Loaded weights: {wpath}")
    else:
        print(f"[warn] Weights not found; using random initialization: {wpath}")
    return model


def compute_prnet_paper_papr(model: PRNet, real_symbols: np.ndarray) -> np.ndarray:
    paper_proxy = []
    chunk = 2000

    for i in range(0, len(real_symbols), chunk):
        _, x_time = model.encode(real_symbols[i : i + chunk], training=False)
        x_np = x_time.numpy()
        paper_proxy.append(compute_papr_amplitude_proxy_db(x_np))

    return np.concatenate(paper_proxy)


def main(num_symbols: int, oversampling: int, seed: int, show_plot: bool) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    np.random.seed(seed)

    model = load_model(oversampling)

    print(f"  Generating {num_symbols:,} random OFDM symbols per curve ...")
    _, symbols = generate_qpsk_symbols(num_symbols, N)
    real_symbols = symbols_to_real(symbols).astype(np.float32)

    print(f"  Computing PAPR - Original OFDM, standard power PAPR (L={oversampling}) ...")
    papr_orig = compute_papr_db(ofdm_ifft(symbols, oversampling))

    print(f"  Computing PAPR - PRNet, paper-compatible peak-amplitude metric (L={oversampling}) ...")
    papr_prnet = compute_prnet_paper_papr(model, real_symbols)

    print("\n  Approximate PAPR0 at CCDF=1e-3")
    print(f"    Original OFDM             : {percentile_at_ccdf(papr_orig):.3f} dB")
    print(f"    PRNet paper-compatible    : {percentile_at_ccdf(papr_prnet):.3f} dB")
    print("    Note: the PRNet curve matches the paper/code metric; it is not standard power PAPR.")

    ccdf_orig = compute_ccdf(papr_orig, PAPR_THRESHOLDS)
    ccdf_prnet = compute_ccdf(papr_prnet, PAPR_THRESHOLDS)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.semilogy(PAPR_THRESHOLDS, ccdf_orig, "k--", lw=2, label="Original OFDM")
    ax.semilogy(PAPR_THRESHOLDS, ccdf_prnet, "r-", lw=2.5, label="PRNet")

    ax.set_xlabel("PAPR0 [dB]", fontsize=13)
    ax.set_ylabel("CCDF (PAPR > PAPR0)", fontsize=13)
    ax.set_title("CCDF of PAPR - Rayleigh-Trained PRNet")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_xlim([0, 14])
    ax.set_ylim([1e-3, 1.1])

    plt.tight_layout()

    out = os.path.join(FIG_DIR, "ccdf_papr_rayleigh.jpg")
    plt.savefig(out, dpi=200)
    print(f"\n[ok] Saved -> {out}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-symbols", type=int, default=NUM_SYMBOLS)
    parser.add_argument(
        "--oversampling",
        type=int,
        default=CCDF_OVERSAMPLING,
        help="IFFT oversampling factor for CCDF evaluation. Default 1 matches the authors' released PRNet CCDF code.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducible CCDF readings.")
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window.")
    args = parser.parse_args()
    main(args.num_symbols, args.oversampling, args.seed, not args.no_show)
