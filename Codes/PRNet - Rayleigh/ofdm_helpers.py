# ═══════════════════════════════════════════════════════════════════════════
# ofdm_helpers.py — OFDM Utility Functions
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Core building blocks for OFDM (Orthogonal Frequency Division
#          Multiplexing) signal processing and simulation
#
# FEATURES:
# - QPSK Modulation/Demodulation: Convert bits to symbols and vice versa
# - OFDM IFFT/FFT: Transform between frequency and time domains
# - PAPR Calculation: Measure Peak-to-Average Power Ratio
# - CCDF Computation: Statistical analysis of PAPR distribution
# - Rayleigh Flat Fading Channel: Frequency-domain fading + AWGN + ZF equalization
# - AWGN Channel: Custom additive white Gaussian noise (no CommPy dependency)
# - BER Calculation: Measure communication quality
#
# LIBRARIES USED:
# - NumPy: For fast FFT/IFFT, array operations, and channel simulation
#
# REFERENCE:
# M. Kim, W. Lee, D.-H. Cho, "A Novel PAPR Reduction Scheme for OFDM
# System based on Deep Learning," IEEE Commun. Lett., 2017.
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 1: Import Required Libraries
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Import the necessary Python libraries for OFDM signal processing
#
# - numpy: For array operations, mathematical functions (FFT/IFFT), and
#          channel simulation (AWGN + Rayleigh fading)
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 2: QPSK Constellation Mapping Tables
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Define lookup tables for fast QPSK modulation and demodulation
#
# QPSK (Quadrature Phase Shift Keying) maps 2 bits to 1 complex symbol.
# We use Gray coding to minimize bit errors when symbols are misdetected.
#
# _QPSK_TABLE: Maps bit pairs to complex symbols (diagonal Gray-coded QPSK)
#   - [0,0] → +1/√2 + 1j/√2  (45°,  1st quadrant)
#   - [0,1] → -1/√2 + 1j/√2  (135°, 2nd quadrant)
#   - [1,0] → +1/√2 - 1j/√2  (315°, 4th quadrant)
#   - [1,1] → -1/√2 - 1j/√2  (225°, 3rd quadrant)
#
# _QPSK_BITS: Reverse mapping for demodulation (symbol index -> bit pair)
# ═══════════════════════════════════════════════════════════════════════════

# QPSK constellation mapping — diagonal Gray-coded (symbols at 45°/135°/225°/315°):
# Index calculation: index = b0*2 + b1
#   index 0: [0,0] → +1/√2 + 1j/√2  (45°,  1st quadrant)
#   index 1: [0,1] → -1/√2 + 1j/√2  (135°, 2nd quadrant)
#   index 2: [1,0] → +1/√2 - 1j/√2  (315°, 4th quadrant)
#   index 3: [1,1] → -1/√2 - 1j/√2  (225°, 3rd quadrant)
_INV_SQRT2 = np.float32(1.0 / np.sqrt(2))   # ≈ 0.70711
_QPSK_TABLE = np.array(
    [+_INV_SQRT2 + 1j*_INV_SQRT2,
     -_INV_SQRT2 + 1j*_INV_SQRT2,
     +_INV_SQRT2 - 1j*_INV_SQRT2,
     -_INV_SQRT2 - 1j*_INV_SQRT2], dtype=np.complex64
)

# Reverse mapping for demodulation: index → bit pair
_QPSK_BITS = np.array(
    [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32
)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 3: QPSK Modulation Function
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Convert binary bits to QPSK complex symbols
#
# PROCESS:
# 1. Reshape the input bit array to group every 2 consecutive bits
# 2. Convert each bit pair [b0, b1] to an index: b0*2 + b1
# 3. Look up the corresponding complex symbol from _QPSK_TABLE
#
# EXAMPLE:
#   Input bits:  [0,0,1,1,0,1,1,0]
#   Pairs:       [[0,0], [1,1], [0,1], [1,0]]
#   Indices:     [0, 3, 1, 2]
#   Symbols:     [+1+0j, -1+0j, 0+1j, 0-1j]
#
# This is fully vectorized (no loops) for maximum performance.
# ═══════════════════════════════════════════════════════════════════════════

def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    # Map bits to QPSK symbols using a vectorized lookup table.
    #
    # Each pair of bits (b0, b1) is converted to a Gray-coded index
    # via index = b0*2 + b1, then looked up in _QPSK_TABLE.
    # Fully vectorized — no Python for-loop.
    #
    # Parameters
    # ----------
    # bits : ndarray, shape (..., 2*N) — binary data (int)
    #
    # Returns
    # -------
    # symbols : ndarray, shape (..., N), complex64
    # Reshape so the last axis is (N, 2) — pairs of bits per subcarrier
    b = bits.reshape(*bits.shape[:-1], -1, 2)   # (..., N, 2)
    # Gray-code index: b0*2 + b1  →  0,1,2,3
    idx = b[..., 0] * 2 + b[..., 1]             # (..., N)
    return _QPSK_TABLE[idx]                       # (..., N) complex64


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 4: QPSK Demodulation Function
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Convert received QPSK symbols back to binary bits
#
# PROCESS:
# 1. For each received symbol, compute its distance to all 4 QPSK points
# 2. Find the closest constellation point (minimum distance)
# 3. Look up the bit pair corresponding to that constellation point
# 4. Flatten the result back to a 1D bit array
#
# This is "hard-decision" demodulation: we simply pick the nearest symbol.
# No soft information is used (unlike in advanced decoders).
#
# EXAMPLE:
#   Received symbols (with noise): [+0.9+0.1j, -0.8-0.2j]
#   Closest points:                [+1+0j,     -1+0j]
#   Output bits:                   [0,0,       1,1]
# ═══════════════════════════════════════════════════════════════════════════

def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    # Hard-decision QPSK demodulation via nearest-neighbour lookup.
    #
    # Fully vectorized — no Python for-loop.
    #
    # Parameters
    # ----------
    # symbols : ndarray, shape (..., N), complex
    #
    # Returns
    # -------
    # bits : ndarray, shape (..., 2*N), int32
    # Compute squared Euclidean distance to every constellation point
    # symbols: (..., N)  →  expand to (..., N, 1) for broadcasting
    diff = symbols[..., np.newaxis] - _QPSK_TABLE   # (..., N, 4)
    dist = np.abs(diff) ** 2                         # (..., N, 4) real
    # Index of closest constellation point per subcarrier
    idx  = np.argmin(dist, axis=-1)                  # (..., N)
    # Look up the 2-bit representation and flatten last two axes
    bits_out = _QPSK_BITS[idx]                       # (..., N, 2)
    return bits_out.reshape(*symbols.shape[:-1], -1).astype(np.int32)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 5: OFDM Inverse Fast Fourier Transform (IFFT)
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Convert frequency-domain symbols to time-domain signal
#
# OFDM CONCEPT:
# - OFDM transmits data on multiple parallel subcarriers (N carriers)
# - Each subcarrier carries one QPSK symbol
# - IFFT transforms these N frequency symbols into N time samples
# - Oversampling (L=4): We zero-pad to N*L before IFFT for better signal representation
#
# PROCESS:
# 1. Zero-pad the input from N symbols to N*L symbols
# 2. Apply Inverse FFT to get time-domain signal
#
# WHY OVERSAMPLE?
# - Oversampling (L>1) gives a more accurate representation of the continuous signal
# - Required to properly compute Peak-to-Average Power Ratio (PAPR)
# ═══════════════════════════════════════════════════════════════════════════

def ofdm_ifft(X: np.ndarray, L: int = 4) -> np.ndarray:
    # Oversampled IFFT: zero-pad *X* from N to N·L, then np.fft.ifft.
    #
    # Parameters
    # ----------
    # X : ndarray, shape (..., N), complex
    # L : int — oversampling factor (default 4)
    #
    # Returns
    # -------
    # x : ndarray, shape (..., N*L), complex
    N = X.shape[-1]
    # Create zero-padded array (N*L total samples)
    padded = np.zeros((*X.shape[:-1], N * L), dtype=np.complex64)
    # Copy input symbols to first N positions
    padded[..., :N] = X
    # Apply Inverse FFT to get time-domain signal
    return np.fft.ifft(padded, axis=-1).astype(np.complex64)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 6: OFDM Fast Fourier Transform (FFT)
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Convert received time-domain signal back to frequency-domain symbols
#
# RECEIVER OPERATION:
# 1. Apply FFT to the received time-domain signal
# 2. Extract only the first N subcarriers (discard zero-padded part)
# 3. Each output element is the received symbol for that subcarrier
#
# This is the inverse operation of ofdm_ifft:
#   Transmitter: symbols --[IFFT]--> time signal --[channel]-->
#   Receiver:    received signal --[FFT]--> symbols
# ═══════════════════════════════════════════════════════════════════════════

def ofdm_fft(y: np.ndarray, N: int) -> np.ndarray:
    # FFT and keep the first *N* subcarriers.
    #
    # Parameters
    # ----------
    # y : ndarray, shape (..., N*L), complex
    # N : int — number of data subcarriers
    #
    # Returns
    # -------
    # Y : ndarray, shape (..., N), complex
    return np.fft.fft(y, axis=-1)[..., :N].astype(np.complex64)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 7: Peak-to-Average Power Ratio (PAPR) Calculation
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Measure how "peaky" the OFDM signal is
#
# PAPR PROBLEM:
# - OFDM signals have high peak power compared to average power
# - High PAPR requires expensive power amplifiers with large linear range
# - If amplifier saturates, signal gets distorted → bit errors
#
# CALCULATION:
#   PAPR = Peak Power / Average Power
#   PAPR (dB) = 10 * log10(max(|x|²) / mean(|x|²))
#
# GOAL OF PRNet:
# - Reduce PAPR while maintaining bit error rate performance
# - Lower PAPR = cheaper, more efficient transmitter hardware
# ═══════════════════════════════════════════════════════════════════════════

def compute_papr_db(x: np.ndarray) -> np.ndarray:
    # PAPR in dB for each OFDM symbol in *x*.
    #
    # Parameters
    # ----------
    # x : ndarray, shape (batch, N*L), complex
    #
    # Returns
    # -------
    # papr : ndarray, shape (batch,)
    # Compute instantaneous power: |x(t)|²
    power = np.abs(x) ** 2
    # Find peak power across all time samples
    peak  = np.max(power, axis=-1)
    # Compute average power across all time samples
    mean  = np.mean(power, axis=-1)
    # PAPR = peak/average, converted to dB scale
    return 10.0 * np.log10(peak / (mean + 1e-12))


def compute_papr_amplitude_proxy_db(x: np.ndarray) -> np.ndarray:
    """Paper-compatible PRNet proxy, not the physical PAPR definition.

    The authors' released PRNet code stores max(abs(x)) after normalizing the
    transmitted waveform and then plots 10*log10(...) for the CCDF. That is an
    amplitude metric, equal to one half of the usual PAPR in dB when the mean
    signal power is normalized. Keep compute_papr_db() for scientific results;
    use this only when reproducing the paper's Fig. 3 PRNet curve.
    """
    power = np.abs(x) ** 2
    mean = np.mean(power, axis=-1, keepdims=True)
    x_norm = x / np.sqrt(mean + 1e-12)
    peak_amp = np.max(np.abs(x_norm), axis=-1)
    return 10.0 * np.log10(peak_amp + 1e-12)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 8: Complementary Cumulative Distribution Function (CCDF) for PAPR
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Compute the probability that PAPR exceeds various threshold values
#
# CCDF DEFINITION:
#   CCDF(threshold) = Probability(PAPR > threshold)
#
# USAGE:
# - Used to characterize PAPR distribution statistically
# - Shows what percentage of OFDM symbols have PAPR above a certain level
# - Lower CCDF curve = better PAPR reduction
#
# EXAMPLE:
#   If CCDF(8 dB) = 0.001, then only 0.1% of symbols have PAPR > 8 dB
#
# This function computes empirical CCDF from observed PAPR values.
# ═══════════════════════════════════════════════════════════════════════════

def compute_ccdf(papr_values: np.ndarray,
                 thresholds: np.ndarray) -> np.ndarray:
    # Empirical CCDF: Prob(PAPR > threshold) for each threshold.
    #
    # Parameters
    # ----------
    # papr_values : ndarray, shape (num_symbols,) — PAPR in dB
    # thresholds  : ndarray, shape (T,) — PAPR₀ values in dB
    #
    # Returns
    # -------
    # ccdf : ndarray, shape (T,)
    return np.array([np.mean(papr_values > t) for t in thresholds])


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 9: AWGN (Additive White Gaussian Noise) Channel Simulation
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Simulate wireless channel by adding noise to transmitted signal
#
# AWGN CHANNEL:
# - Simplest channel model: y = x + noise
# - "White" = noise has equal power at all frequencies
# - "Gaussian" = noise amplitude follows normal distribution
#
# IMPLEMENTATION:
# - Custom NumPy implementation (no external dependency)
# - Noise power is calibrated from the measured signal power and SNR:
#     noise_var = signal_power / (10^(SNR_dB/10))
# - Complex noise: both real and imaginary parts are i.i.d. Gaussian
#   with variance noise_var/2 each (total variance = noise_var)
#
# SNR (Signal-to-Noise Ratio):
# - Measures how much stronger the signal is compared to noise
# - Higher SNR = less noise = fewer bit errors
# - SNR in dB: 10*log10(signal power / noise power)
#
# TYPICAL VALUES:
# - SNR = 0 dB:  very noisy (signal = noise power)
# - SNR = 20 dB: clean signal (signal 100x stronger than noise)
# ═══════════════════════════════════════════════════════════════════════════

def awgn_channel(x: np.ndarray, snr_db: float, L: int = 1) -> np.ndarray:
    # Custom AWGN channel (no CommPy dependency).
    #
    # Parameters
    # ----------
    # x      : ndarray, complex — transmitted signal
    # snr_db : float — target SNR in dB for the active subcarriers
    # L      : int — oversampling factor (default 1). Required to scale noise
    #                correctly if x is an oversampled time-domain signal.
    #
    # Returns
    # -------
    # y : ndarray, complex — received signal  (x + noise)
    sig_pow = np.mean(np.abs(x) ** 2)
    
    # SCIENTIFIC CORRECTION: Adjust noise variance for oversampling. 
    # When the receiver discards the out-of-band frequencies (via FFT slicing), 
    # it filters out (L-1)/L of the noise. To maintain the exact target SNR 
    # within the active subcarrier band, we must inject L times more noise.
    noise_var = (sig_pow / (10.0 ** (snr_db / 10.0))) * L
    
    std = np.sqrt(noise_var / 2.0)
    noise = (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)) * std
    return (x + noise).astype(np.complex64)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 9b: Rayleigh Flat Fading Channel (Frequency-Domain)
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Simulate a flat Rayleigh fading wireless channel for OFDM
#
# RAYLEIGH FLAT FADING CHANNEL (per-subcarrier model):
#   Y_k = H_k · X_k + N_k
#
# where:
#   H_k ~ CN(0, 1)  — i.i.d. complex Gaussian fading coefficient
#   N_k             — AWGN noise calibrated to the desired SNR
#
# WHY FREQUENCY DOMAIN?
# - In OFDM with a cyclic prefix, each subcarrier sees an independent
#   flat fading coefficient. This is the standard model and is
#   computationally efficient (element-wise multiply, no convolution).
#
# ZERO-FORCING (ZF) EQUALIZATION at the receiver:
#   X̂_k = Y_k / H_k
# - Assumes perfect Channel State Information (CSI).
# - This is the standard "ideal receiver" assumption used in the paper.
#
# REFERENCE:
# Kim et al., 2017: "We also assume the Rayleigh fading channel for
# a wireless channel, i.e., H." (Section IV)
# ═══════════════════════════════════════════════════════════════════════════

def rayleigh_flat_channel(X_freq: np.ndarray,
                          snr_db: float) -> tuple:
    # Rayleigh flat fading channel with ZF equalization.
    #
    # Operates in the frequency domain (per-subcarrier):
    #   1. Generate H ~ CN(0,1) per subcarrier
    #   2. Y = H * X + N  (fading + noise)
    #   3. X̂ = Y / H      (ZF equalization, perfect CSI)
    #
    # Parameters
    # ----------
    # X_freq : ndarray, shape (batch, N), complex
    #          Frequency-domain OFDM symbols (encoded subcarrier values)
    # snr_db : float — SNR in dB
    #
    # Returns
    # -------
    # Y_eq : ndarray, shape (batch, N), complex
    #        ZF-equalized received symbols
    # H    : ndarray, shape (batch, N), complex
    #        Channel coefficients (for analysis/debugging)

    # 1. Rayleigh fading: H ~ CN(0, 1)
    #    |H| is Rayleigh-distributed, phase is uniform
    H = (np.random.randn(*X_freq.shape)
         + 1j * np.random.randn(*X_freq.shape)) / np.sqrt(2.0)

    # 2. Apply fading: Y_faded = H * X
    Y_faded = H * X_freq

    # 3. Add AWGN noise calibrated to SNR
    #    Noise power computed from *faded* signal power
    sig_pow = np.mean(np.abs(Y_faded) ** 2)
    noise_var = sig_pow / (10.0 ** (snr_db / 10.0))
    std = np.sqrt(noise_var / 2.0)
    noise = (np.random.randn(*X_freq.shape)
             + 1j * np.random.randn(*X_freq.shape)) * std
    Y = Y_faded + noise

    # 4. Zero-Forcing equalization: X̂ = Y / H
    Y_eq = Y / H

    return Y_eq.astype(np.complex64), H.astype(np.complex64)



# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 10: Random QPSK Data Generation
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Create random test data for training and evaluation
#
# PROCESS:
# 1. Generate random binary bits (0s and 1s)
# 2. Modulate them to QPSK symbols using qpsk_modulate()
# 3. Return both bits and symbols
#
# USAGE:
# - For training: generate fresh random data for each batch
# - For testing: generate fixed test set to measure performance
#
# PARAMETERS:
# - num_symbols: how many OFDM symbols to generate
# - N: number of subcarriers per symbol (default 64)
# - Total bits generated: num_symbols * 2*N
# ═══════════════════════════════════════════════════════════════════════════

def generate_qpsk_symbols(num_symbols: int, N: int = 64):
    # Generate random QPSK data.
    #
    # Returns
    # -------
    # bits    : ndarray, shape (num_symbols, 2*N) — original bits
    # symbols : ndarray, shape (num_symbols, N), complex
    bits = np.random.randint(0, 2, size=(num_symbols, 2 * N)).astype(np.int32)
    symbols = qpsk_modulate(bits)
    return bits, symbols


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 11: Complex-to-Real Conversion for Neural Networks
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Convert complex QPSK symbols to real-valued vectors for neural network input
#
# WHY?
# - Neural networks (like PRNet) work with real numbers, not complex numbers
# - We need to represent complex symbols as real vectors
#
# CONVERSION:
#   Complex symbol:  a + bj
#   Real vector:     [a, b]  (real part, imaginary part)
#
# EXAMPLE:
#   Input:  [+1+0j, 0+1j, -1+0j]  (3 complex symbols)
#   Output: [+1, 0,  0, +1,  -1, 0]  (6 real numbers)
#
# symbols_to_real(): complex → real (for network input)
# real_to_symbols(): real → complex (for network output)
# ═══════════════════════════════════════════════════════════════════════════

def symbols_to_real(symbols: np.ndarray) -> np.ndarray:
    # Stack Re/Im of complex symbols → real vector.
    #
    # (batch, N) complex  →  (batch, 2N) real
    # Separate real and imaginary parts, then interleave them
    return np.stack([symbols.real, symbols.imag], axis=-1) \
             .reshape(*symbols.shape[:-1], -1).astype(np.float32)


def real_to_symbols(real_vec: np.ndarray) -> np.ndarray:
    # Inverse of *symbols_to_real*.
    #
    # (batch, 2N) real  →  (batch, N) complex
    # Reshape to separate real/imag pairs
    r = real_vec.reshape(*real_vec.shape[:-1], -1, 2)
    # Reconstruct complex numbers: a + bj
    return (r[..., 0] + 1j * r[..., 1]).astype(np.complex64)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 12: Bit Error Rate (BER) Calculation
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Measure communication quality by counting bit errors
#
# BER DEFINITION:
#   BER = (Number of bit errors) / (Total number of bits)
#
# EXAMPLE:
#   Transmitted: [0, 1, 0, 1, 1, 0, 1, 0]
#   Received:    [0, 1, 1, 1, 1, 0, 0, 0]
#                         ↑           ↑ ↑  (3 errors)
#   BER = 3/8 = 0.375
#
# TYPICAL VALUES:
# - BER = 10⁻³ (0.001):  1 error per 1000 bits (acceptable for some apps)
# - BER = 10⁻⁶ (0.000001): 1 error per million bits (good quality)
# - BER = 10⁻⁹:  very high quality
#
# Lower BER = better system performance
# ═══════════════════════════════════════════════════════════════════════════

def compute_ber(bits_tx: np.ndarray, bits_rx: np.ndarray) -> float:
    # Bit Error Rate between transmitted and received bit arrays.
    return np.mean(bits_tx != bits_rx)
