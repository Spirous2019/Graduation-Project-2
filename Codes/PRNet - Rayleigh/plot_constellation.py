# ═══════════════════════════════════════════════════════════════════════════
# plot_constellation.py — PRNet Encoder Output Constellation Diagram
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Visualise how the PRNet encoder modifies QPSK symbols
#
# WHAT THIS SHOWS:
# - PRNet encoder output — symbols repositioned to reduce PAPR
# - Points are color-coded by their original QPSK symbol (4 classes)
#
# NOTE: This is the Rayleigh version. The encoder output may differ from
# the AWGN version since this model was trained with Rayleigh fading.
#
# USAGE:
#   python plot_constellation.py
# ═══════════════════════════════════════════════════════════════════════════

import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prnet_model import PRNet
from ofdm_helpers import generate_qpsk_symbols, symbols_to_real

# ── Parameters ────────────────────────────────────────────────────────────
N               = 64
NUM_SYMBOLS     = 200         # OFDM symbols to plot (200 × 64 = 12,800 subcarrier points)
LAMBDA          = 0.01
MODEL_DIR       = os.path.join(os.path.dirname(__file__), "models")
FIG_DIR         = os.path.join(os.path.dirname(__file__), "figures")




def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # ── Load trained PRNet ────────────────────────────────────────────
    print("Loading PRNet model...")
    model = PRNet(N=N, num_blocks=5, hidden_dim=2048,
                  lambda_=LAMBDA, eta_db=-15.0)
    model(tf.zeros((1, 2 * N)), training=False)          # build

    wpath = os.path.join(MODEL_DIR, f"prnet_lambda_{LAMBDA}.weights.h5")
    if not os.path.exists(wpath):
        print(f"[error] Weights not found: {wpath}")
        return
    model.load_weights(wpath)
    print(f"[ok] Loaded weights: {wpath}")

    # ── Generate proper QPSK input ───────────────────────────────────
    _, symbols = generate_qpsk_symbols(NUM_SYMBOLS, N)
    r = symbols_to_real(symbols).astype(np.float32)

    print(f"Generating constellation for {NUM_SYMBOLS} OFDM symbols "
          f"({NUM_SYMBOLS * N:,} subcarrier points) ...")

    # ── Encode ────────────────────────────────────────────────────────
    X_enc, _ = model.encode(r, training=False)

    # Flatten: (NUM_SYMBOLS, N) → (NUM_SYMBOLS*N,)
    sym_flat = symbols.reshape(-1)
    enc_flat = tf.reshape(X_enc, [-1]).numpy()

    # Print scale info
    print(f"  Input  scale: Re in [{sym_flat.real.min():.3f}, {sym_flat.real.max():.3f}]")
    print(f"  Output scale: Re in [{enc_flat.real.min():.3f}, {enc_flat.real.max():.3f}]")



    # ── Plot: Constellation ───────────────────────────────────────────
    fig, ax2 = plt.subplots(figsize=(7, 6.5))

    # 4 QPSK source classes and their display colors
    # IMPORTANT: use float32 so that (sym_flat == qp) exact-equality works —
    # sym_flat comes from _QPSK_TABLE (complex64) and must match bit-for-bit.
    _S = np.float32(1.0 / np.sqrt(2))
    QPSK_PTS    = list(np.array([+_S+1j*_S, -_S+1j*_S,
                                  +_S-1j*_S, -_S-1j*_S], dtype=np.complex64))
    # 4 QPSK source classes — original seaborn palette
    CLS_COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    CLS_LABELS  = ["(+,\u2009+)", "(−,\u2009+)", "(+,\u2009−)", "(−,\u2009−)"]   # thin-space, unicode −
    MAX_PER_CLS = 1000   # up to 1000 points per class

    # Plot each class cloud in its own color
    for qp, color, lbl in zip(QPSK_PTS, CLS_COLORS, CLS_LABELS):
        mask    = (sym_flat == qp)
        enc_cls = enc_flat[mask]
        if len(enc_cls) > MAX_PER_CLS:
            pick = np.random.choice(len(enc_cls), MAX_PER_CLS, replace=False)
            enc_cls = enc_cls[pick]
        ax2.scatter(enc_cls.real, enc_cls.imag,
                    c=color, alpha=0.45, s=8, label=lbl)

    # QPSK reference crosses at the diagonal symbol locations (±1/√2, ±1/√2)
    ref = np.array(QPSK_PTS)
    ax2.scatter(ref.real, ref.imag,
                marker="+", s=120, linewidths=2.0,
                c="#0A3D91", zorder=6, label=r"QPSK inputs $s_k$")

    ax2.axhline(0, color="grey", lw=0.5)
    ax2.axvline(0, color="grey", lw=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("In-Phase", fontsize=12)
    ax2.set_ylabel("Quadrature", fontsize=12)
    ax2.set_aspect("equal")
    ax2.legend(fontsize=8, loc="upper right", markerscale=1.0)

    fig.suptitle(f"PRNet Encoder Constellation (Rayleigh Channel, λ={LAMBDA})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "encoder_constellation_rayleigh.jpg")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\n[ok] Saved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
