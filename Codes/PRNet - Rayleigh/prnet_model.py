# ═══════════════════════════════════════════════════════════════════════════
# prnet_model.py — PRNet Keras Model
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Deep autoencoder neural network for PAPR reduction in OFDM systems
#
# ARCHITECTURE (from Kim et al., 2017, Section III):
#   Encoder: Input(2N) → [FC(2048)→BN→ReLU] × 5 → FC(2N, linear) → Power Norm
#   Decoder: Input(2N) → [FC(2048)→BN→ReLU] × 5 → FC(2N, linear)
#
# COMPLETE SIGNAL PATH:
#   Original symbols (r)
#   → Encoder: Modify symbols to reduce PAPR
#   → Power Normalization: Scale to match input power (prevents energy explosion)
#   → X_enc: Modified frequency-domain symbols (unit average power)
#   → IFFT: Convert to time-domain signal (low PAPR!)
#   → Channel: Rayleigh flat fading + AWGN noise (only during training)
#   → ZF Equalization: Remove fading (Y/H)
#   → Decoder: Recover original symbols
#   → r̂: Reconstructed symbols
#
# KEY INNOVATION:
# - Joint training with loss = MSE(r, r̂) + λ*PAPR(time_signal)
# - Network learns to minimize PAPR while preserving data fidelity
#
# REFERENCE:
# M. Kim, W. Lee, D.-H. Cho, "A Novel PAPR Reduction Scheme for OFDM
# System based on Deep Learning," IEEE Commun. Lett., 2017.
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 1: Import Deep Learning Libraries
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Import TensorFlow/Keras for building the neural network
#
# - tensorflow: Deep learning framework for building and training PRNet
# - numpy: For numerical array operations
# ═══════════════════════════════════════════════════════════════════════════

import tensorflow as tf
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 2: Build Deep Neural Network (Encoder/Decoder)
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Construct the encoder or decoder sub-network architecture
#
# ARCHITECTURE:
#   Input → [Dense(2048) → BatchNorm → ReLU] × 5 → Dense(output, linear)
#
# WHY THIS DESIGN?
# - Dense (Fully Connected) layers: Learn complex non-linear transformations
# - BatchNormalization: Stabilizes training by normalizing layer inputs
# - ReLU activation: Introduces non-linearity, enables learning complex patterns
# - 5 hidden blocks: Provides sufficient depth to learn PAPR reduction
# - Linear output: Allows network to output any real values (symbols can be any complex number)
#
# This function is called twice:
#   1. build_dnn for Encoder (modifies symbols to reduce PAPR)
#   2. build_dnn for Decoder (recovers original symbols after channel)
# ═══════════════════════════════════════════════════════════════════════════

def build_dnn(input_dim: int,
              output_dim: int,
              num_blocks: int = 5,
              hidden_dim: int = 2048,
              name: str = "dnn") -> tf.keras.Model:
    # Construct the encoder or decoder sub-network.
    #
    # Each hidden block:  Dense → BatchNormalization → ReLU
    # Output block:       Dense (linear)
    # Input layer
    inp = tf.keras.Input(shape=(input_dim,), name=f"{name}_input")
    x = inp

    # Hidden blocks (repeated num_blocks times)
    for i in range(num_blocks):
        # Fully-connected layer with 2048 neurons
        x = tf.keras.layers.Dense(hidden_dim, use_bias=True,
                                  name=f"{name}_fc{i+1}")(x)
        # Batch normalization for training stability
        x = tf.keras.layers.BatchNormalization(name=f"{name}_bn{i+1}")(x)
        # ReLU activation for non-linearity
        x = tf.keras.layers.ReLU(name=f"{name}_relu{i+1}")(x)

    # Output layer (linear activation allows any real value)
    x = tf.keras.layers.Dense(output_dim, activation=None,
                              name=f"{name}_output")(x)

    return tf.keras.Model(inputs=inp, outputs=x, name=name)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 3: TensorFlow-Compatible Data Conversion Functions
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Convert between real and complex representations inside TensorFlow graph
#
# WHY TENSORFLOW VERSIONS?
# - These operate on TensorFlow tensors (not NumPy arrays)
# - Must be differentiable for backpropagation during training
# - Allow gradients to flow through the entire network (encoder → IFFT → FFT → decoder)
#
# FLOW:
#   Network output (real) → complex → IFFT → time signal → FFT → complex → real → Network input
# ═══════════════════════════════════════════════════════════════════════════

def tf_real_to_complex(real_vec):
    # (batch, 2N) real → (batch, N) complex.
    # Reshape to separate real/imaginary pairs: (batch, N, 2)
    r = tf.reshape(real_vec, [tf.shape(real_vec)[0], -1, 2])
    # Construct complex numbers from real and imaginary components
    return tf.complex(r[:, :, 0], r[:, :, 1])


def tf_complex_to_real(symbols):
    # (batch, N) complex → (batch, 2N) real.
    # Extract real and imaginary parts separately
    re = tf.math.real(symbols)
    im = tf.math.imag(symbols)
    # Stack and flatten: [re, im, re, im, ...] → (batch, 2N)
    return tf.reshape(tf.stack([re, im], axis=-1),
                      [tf.shape(symbols)[0], -1])


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 4: TensorFlow OFDM Operations (Differentiable IFFT/FFT)
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Implement OFDM transformations that work within TensorFlow's autodiff
#
# CRITICAL FOR TRAINING:
# - These functions are differentiable (gradients can flow through them)
# - Allows the network to learn how encoder changes affect PAPR via backpropagation
#
# tf_ofdm_ifft: frequency → time (creates the actual transmitted signal)
# tf_ofdm_fft: time → frequency (receiver operation)
# ═══════════════════════════════════════════════════════════════════════════

def tf_ofdm_ifft(X, L=4):
    # Oversampled IFFT:  zero-pad then IFFT.
    #
    # (batch, N) complex → (batch, N*L) complex
    N = tf.shape(X)[-1]
    # Create zero padding to oversample by factor L
    padding = tf.zeros([tf.shape(X)[0], N * (L - 1)], dtype=X.dtype)
    # Concatenate input with zeros: [X, 0, 0, 0, ...] for L=4
    X_padded = tf.concat([X, padding], axis=-1)          # (batch, N*L)
    # Apply Inverse FFT to get oversampled time-domain signal
    return tf.signal.ifft(X_padded)


def tf_ofdm_fft(y, N):
    # FFT and keep first N subcarriers.
    #
    # (batch, N*L) complex → (batch, N) complex
    # Apply FFT to entire signal and extract first N frequency bins
    return tf.signal.fft(y)[:, :N]


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 5: Differentiable PAPR Computation
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Calculate PAPR inside TensorFlow for use as a loss function
#
# WHY DIFFERENTIABLE PAPR?
# - This function is used in the training loss: Loss = MSE + λ*PAPR
# - Network learns to minimize PAPR by gradient descent
# - Returns PAPR in linear scale (not dB) for smoother gradients
#
# KEY INSIGHT:
# - By including PAPR in the loss, the encoder learns to shape symbols
#   such that their IFFT produces low-PAPR time-domain signals
# ═══════════════════════════════════════════════════════════════════════════

def tf_papr(x_time):
    # Differentiable PAPR (linear scale, not dB).
    #
    # Parameters
    # ----------
    # x_time : complex tensor, shape (batch, N*L)
    #
    # Returns
    # -------
    # papr : real tensor, shape (batch,)
    # Compute instantaneous power: |x(t)|²
    power = tf.cast(tf.abs(x_time) ** 2, tf.float32)
    # Find peak power value across time
    peak  = tf.reduce_max(power, axis=-1)                # (batch,)
    # Compute average power across time
    mean  = tf.reduce_mean(power, axis=-1)               # (batch,)
    # PAPR = peak / average (linear scale, not dB for gradient smoothness)
    return peak / (mean + 1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 6: PRNet Main Class - Complete PAPR Reduction Autoencoder
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: End-to-end trainable system for PAPR reduction in OFDM
#
# COMPLETE SIGNAL FLOW:
#   1. Input: Original QPSK symbols (as real vector)
#   2. Encoder: Modifies symbols to reduce PAPR
#   3. IFFT: Convert to time-domain signal (this is where PAPR matters!)
#   4. Channel: Rayleigh flat fading + AWGN noise (only during training)
#   5. ZF Equalization: Remove fading (Ŷ = Y/H)
#   6. Decoder: Recover original symbols
#   7. Output: Reconstructed symbols
#
# TRAINING STRATEGY:
#   - Loss = MSE(original, reconstructed) + λ * PAPR(time_signal)
#   - λ controls trade-off: smaller λ → better BER, larger λ → lower PAPR
#   - Training with Rayleigh fading + corruption (η) makes system robust
#
# KEY PARAMETERS:
#   - N: number of subcarriers (64)
#   - L: oversampling factor (4)
#   - lambda_: PAPR loss weight (0.01 = good balance)
#   - eta_db: training corruption level (-15 dB = optimal per paper)
# ═══════════════════════════════════════════════════════════════════════════

class PRNet(tf.keras.Model):
    # End-to-end PRNet: Encoder → IFFT → Rayleigh Channel → ZF Eq → Decoder.
    #
    # Parameters
    # ----------
    # N          : int   — number of OFDM subcarriers (default 64)
    # L          : int   — oversampling factor (default 4)
    # num_blocks : int   — hidden sub-blocks per encoder/decoder (default 5)
    # hidden_dim : int   — neurons per FC hidden layer (default 2048)
    # lambda_    : float — PAPR loss weight (default 0.01)
    # eta_db     : float — training corruption level in dB (default −15)

    def __init__(self,
                 N: int = 64,
                 L: int = 4,
                 num_blocks: int = 5,
                 hidden_dim: int = 2048,
                 lambda_: float = 0.01,
                 eta_db: float = -15.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.L = L
        self.lambda_ = lambda_
        self.eta_db = eta_db

        dim = 2 * N
        self.encoder = build_dnn(dim, dim, num_blocks, hidden_dim,
                                 name="encoder")
        self.decoder = build_dnn(dim, dim, num_blocks, hidden_dim,
                                 name="decoder")

    # ═════════════════════════════════════════════════════════════════════
    # Power Normalization
    # ═════════════════════════════════════════════════════════════════════
    # PURPOSE: Constrain encoder output power to match input QPSK power
    #
    # WHY THIS IS NECESSARY:
    # - Without normalization, the encoder can inflate symbol amplitudes
    #   to arbitrary values (e.g. ±1500) because PAPR is a *ratio* and
    #   is invariant to uniform scaling.
    # - Larger amplitudes trivially drown the training noise η, making
    #   MSE reconstruction artificially easy.
    # - The normalization forces the encoder to find *geometrically*
    #   clever constellation shapes rather than simply scaling up.
    #
    # OPERATION:
    #   X_norm = X_enc * sqrt(P_target / P_actual)
    #   where P_target = 1.0 (QPSK average power) and
    #         P_actual = mean(|X_enc|²) per symbol.
    #
    # This is fully differentiable: gradients flow through sqrt and
    # division, so the encoder learns under the power constraint.
    # ═════════════════════════════════════════════════════════════════════

    def _normalize_power(self, X_enc, target_power=1.0):
        # Scale complex symbols so per-symbol average power = target_power.
        #
        # Parameters
        # ----------
        # X_enc        : complex tensor, shape (batch, N)
        # target_power : float — desired average power (default 1.0 for QPSK)
        #
        # Returns
        # -------
        # X_norm : complex tensor, shape (batch, N)
        avg_pow = tf.reduce_mean(
            tf.cast(tf.abs(X_enc) ** 2, tf.float32),
            axis=-1, keepdims=True
        )  # (batch, 1)
        scale = tf.sqrt(target_power / (avg_pow + 1e-8))  # (batch, 1)
        return X_enc * tf.cast(scale, X_enc.dtype)

    # ═════════════════════════════════════════════════════════════════════
    # BLOCK 7: Forward Pass - Complete Signal Processing Chain
    # ═════════════════════════════════════════════════════════════════════
    # PURPOSE: Process input symbols through the entire PRNet pipeline
    #
    # STEPS:
    # 1. Encoder: Transform input symbols to reduce PAPR
    # 2. IFFT: Convert to time-domain (where PAPR is measured)
    # 3. Channel: Rayleigh flat fading + AWGN noise during training only
    # 4. ZF Equalization: Remove fading (Ŷ = Y/H)
    # 5. Decoder: Reconstruct original symbols
    #
    # TRAINING vs INFERENCE:
    # - Training: Rayleigh fading + AWGN with corruption level η
    # - Inference: No channel (evaluation scripts add it externally)
    #
    # RAYLEIGH FLAT FADING (frequency-domain, per-subcarrier):
    #   Y_k = H_k · X_k + N_k,   H_k ~ CN(0, 1)
    #   Ŷ_k = Y_k / H_k          (ZF equalization, perfect CSI)
    # ═════════════════════════════════════════════════════════════════════

    def call(self, inputs, training=False):
        # Full forward:  input → encoder → IFFT → Rayleigh ch → ZF eq → decoder.
        #
        # Parameters
        # ----------
        # inputs : float tensor, shape (batch, 2N) — real-valued QPSK data
        #
        # Returns
        # -------
        # r_hat      : float tensor, shape (batch, 2N) — reconstructed data
        # x_time     : complex tensor, shape (batch, N*L) — time-domain signal
        # ENCODER: Transform input symbols to reduce PAPR
        enc_out = self.encoder(inputs, training=training)    # (batch, 2N)
        X_enc = tf_real_to_complex(enc_out)                  # (batch, N)

        # POWER NORMALIZATION: Constrain output to unit average power
        X_enc = self._normalize_power(X_enc)                 # (batch, N)

        # IFFT: Convert to time-domain signal (this is where PAPR matters!)
        x_time = tf_ofdm_ifft(X_enc, self.L)                # (batch, N*L)

        # RAYLEIGH FLAT FADING + AWGN CHANNEL (only during training)
        if training:
            # Convert corruption level η from dB to linear scale
            eta_lin = 10.0 ** (self.eta_db / 10.0)

            # -- Rayleigh fading coefficients --
            # H_k ~ CN(0, 1)  per subcarrier, per sample in the batch
            # Using real + j*imag, each ~ N(0, 1/2), so |H| is Rayleigh
            H = tf.complex(
                tf.random.normal((tf.shape(X_enc)[0], self.N)) / tf.sqrt(2.0),
                tf.random.normal((tf.shape(X_enc)[0], self.N)) / tf.sqrt(2.0)
            )  # (batch, N)

            # -- Apply fading in the frequency domain --
            # Y_faded_k = H_k * X_enc_k
            X_enc_c64 = tf.cast(X_enc, tf.complex64)
            H_c64 = tf.cast(H, tf.complex64)
            Y_faded = H_c64 * X_enc_c64                      # (batch, N)

            # -- Add AWGN noise calibrated to corruption level η --
            sig_pow = tf.reduce_mean(tf.cast(tf.abs(Y_faded)**2, tf.float32))
            noise_var = eta_lin * tf.cast(sig_pow, tf.float32)
            noise_std = tf.sqrt(tf.cast(noise_var / 2, tf.float32))
            noise = tf.complex(
                tf.random.normal(tf.shape(Y_faded)) * noise_std,
                tf.random.normal(tf.shape(Y_faded)) * noise_std
            )
            Y = Y_faded + tf.cast(noise, Y_faded.dtype)      # (batch, N)

            # -- Zero-Forcing equalization: Ŷ_k = Y_k / H_k --
            # Assumes perfect CSI (receiver knows H)
            Y_eq = Y / H_c64                                 # (batch, N)
        else:
            # No channel at inference (evaluation scripts add it externally)
            # Simply go: IFFT → FFT (identity on the first N bins)
            Y_eq = tf.cast(X_enc, tf.complex64)

        # DECODER: Reconstruct original symbols from equalized signal
        dec_in = tf_complex_to_real(Y_eq)                    # (batch, 2N)
        r_hat = self.decoder(dec_in, training=training)      # (batch, 2N)
        return r_hat, x_time

    # ═════════════════════════════════════════════════════════════════════
    # BLOCK 8: Custom Training Step - Joint Loss Optimization
    # ═════════════════════════════════════════════════════════════════════
    # PURPOSE: Train the network to minimize both reconstruction error AND PAPR
    #
    # LOSS FUNCTION (from paper Equation 7):
    #   Total Loss = L1 + λ * L2
    #   where:
    #     L1 = MSE(original_symbols, reconstructed_symbols)  ← data fidelity
    #     L2 = mean(PAPR(time_signal))                       ← PAPR penalty
    #     λ  = trade-off parameter (0.01)
    #
    # WHY TWO LOSSES?
    # - L1 alone: Perfect reconstruction but high PAPR (no benefit)
    # - L2 alone: Low PAPR but corrupted data (useless communication)
    # - L1 + λ*L2: Optimal balance (low PAPR with good data recovery)
    #
    # @tf.function(jit_compile=True): Compiles to optimized XLA for speed
    # ═════════════════════════════════════════════════════════════════════

    @tf.function(jit_compile=True)  # XLA compilation for faster training
    def train_step(self, data):
        r = data  # Input: original QPSK symbols (real-valued)

        # Forward pass with gradient tracking
        with tf.GradientTape() as tape:
            r_hat, x_time = self(r, training=True)

            # L1: Reconstruction loss (MSE) - ensures data can be recovered
            loss_signal = tf.reduce_mean(tf.square(r - r_hat))

            # L2: PAPR loss (mean PAPR across batch) - encourages low PAPR
            loss_papr = tf.reduce_mean(tf_papr(x_time))

            # Joint loss (Equation 7 from paper)
            # Total = data fidelity + λ * PAPR penalty
            loss = loss_signal + self.lambda_ * loss_papr

        # Backpropagation: compute gradients
        grads = tape.gradient(loss, self.trainable_variables)

        # Update weights using optimizer (Adam)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Return metrics for monitoring
        return {"loss": loss,
                "loss_signal": loss_signal,
                "loss_papr": loss_papr}

    # ═════════════════════════════════════════════════════════════════════
    # BLOCK 9: Separate Encoder/Decoder Access (for Evaluation)
    # ═════════════════════════════════════════════════════════════════════
    # PURPOSE: Allow independent use of encoder and decoder during testing
    #
    # WHY SEPARATE?
    # During evaluation, we need to:
    # 1. Use encoder at transmitter
    # 2. Pass signal through external channel simulation
    # 3. Use decoder at receiver
    #
    # This separation allows testing with different channel models
    # without retraining the network.
    # ═════════════════════════════════════════════════════════════════════

    def encode(self, r, training=False):
        # Encoder forward pass:  r → encoder → IFFT → time-domain signal.
        #
        # Returns
        # -------
        # X_enc  : complex, (batch, N)   — encoded frequency-domain
        # x_time : complex, (batch, N*L) — time-domain signal
        # Apply encoder network to input symbols
        enc_out = self.encoder(r, training=training)
        # Convert real representation back to complex
        X_enc = tf_real_to_complex(enc_out)
        # Power normalization (must match call() exactly)
        X_enc = self._normalize_power(X_enc)
        # Generate time-domain signal via IFFT
        x_time = tf_ofdm_ifft(X_enc, self.L)
        return X_enc, x_time

    def decode(self, Y_eq_real, training=False):
        # Decoder forward pass only.
        # Apply decoder network to received symbols
        return self.decoder(Y_eq_real, training=training)

    def get_config(self):
        return {"N": self.N, "L": self.L,
                "num_blocks": self.encoder.layers.__len__(),
                "lambda_": self.lambda_,
                "eta_db": self.eta_db}


# ──────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = PRNet(N=64, hidden_dim=2048, num_blocks=5)
    dummy = tf.random.normal((4, 128))
    r_hat, x_time = model(dummy, training=True)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {r_hat.shape}")
    print(f"Time-domain  : {x_time.shape}")
    model.encoder.summary()
    model.decoder.summary()
