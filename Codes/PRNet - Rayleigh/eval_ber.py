"""Evaluate PRNet BER over a Rayleigh fading channel.

The plot axis is Eb/N0, matching the paper. Internally the channel simulator
uses Es/N0, so QPSK adds 10*log10(2) dB before noise is generated.
"""

import argparse
import os
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ofdm_helpers import (
    generate_qpsk_symbols,
    qpsk_demodulate,
    rayleigh_flat_channel,
    real_to_symbols,
    symbols_to_real,
)
from prnet_model import PRNet


N = 64
L = 4
BITS_PER_SYMBOL = 2
EB_N0_RANGE = np.arange(0, 31, 1)
NUM_SYMBOLS = 25_000
LAMBDA = 0.01

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")


def load_model() -> PRNet:
    model = PRNet(N=N, L=L, num_blocks=5, hidden_dim=2048, lambda_=LAMBDA, eta_db=-15.0)
    model(tf.zeros((1, 2 * N)), training=False)

    weights_path = os.path.join(MODEL_DIR, f"prnet_lambda_{LAMBDA}.weights.h5")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"[ok] Loaded weights from {weights_path}")
    else:
        print(f"[warn] Weights not found at {weights_path}; using random initialization.")
    return model


def eval_prnet_ber(model: PRNet, bits: np.ndarray, symbols: np.ndarray, eb_n0_db: float) -> float:
    es_n0_db = eb_n0_db + 10.0 * np.log10(BITS_PER_SYMBOL)
    errors = 0
    total = bits.size
    chunk = 2000

    for i in range(0, len(symbols), chunk):
        bit_chunk = bits[i : i + chunk]
        sym_chunk = symbols[i : i + chunk]
        r = symbols_to_real(sym_chunk).astype(np.float32)

        x_enc, _ = model.encode(r, training=False)
        y_eq, _ = rayleigh_flat_channel(x_enc.numpy(), es_n0_db)

        y_real = np.stack([y_eq.real, y_eq.imag], axis=-1).reshape(-1, 2 * N).astype(np.float32)
        r_hat = model.decode(y_real, training=False).numpy()
        bits_rx = qpsk_demodulate(real_to_symbols(r_hat))
        errors += int(np.count_nonzero(bit_chunk != bits_rx))

    return errors / total




def main(num_symbols: int, show_plot: bool) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    model = load_model()
    bits, symbols = generate_qpsk_symbols(num_symbols, N)

    ber_prnet = []
    for eb_n0 in EB_N0_RANGE:
        es_n0 = eb_n0 + 10.0 * np.log10(BITS_PER_SYMBOL)
        print(f"  Eb/N0={eb_n0:>2d} dB (Es/N0={es_n0:5.2f} dB) ... ", end="", flush=True)
        ber = eval_prnet_ber(model, bits, symbols, eb_n0)
        ber_prnet.append(ber)
        print(f"PRNet={ber:.4e}")

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.semilogy(EB_N0_RANGE, ber_prnet, "r-", lw=2, label="PRNet (Rayleigh)")

    ax.set_xlabel("$E_b/N_0$ [dB]", fontsize=13)
    ax.set_ylabel("BER", fontsize=13)
    ax.set_title("BER vs. Eb/N0 - PRNet (Rayleigh Channel)")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_xlim([EB_N0_RANGE[0], EB_N0_RANGE[-1]])
    ax.set_ylim([1e-4, 1])

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "ber_vs_snr_rayleigh.jpg")
    plt.savefig(out, dpi=200)
    print(f"\n[ok] Saved -> {out}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-symbols", type=int, default=NUM_SYMBOLS)
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window.")
    args = parser.parse_args()
    main(args.num_symbols, not args.no_show)
