# ═══════════════════════════════════════════════════════════════════════════
# train_prnet.py — PRNet Training
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Complete training pipeline for PRNet PAPR reduction system
#
# TRAINING STRATEGY (from Kim et al., 2017):
# - The model is trained directly using the optimal hyperparameters identified 
#   in the paper's sweet spot:
#   - λ = 0.01: Trade-off between BER and PAPR reduction penalty
#   - η = -15 dB: Optimal corruption noise level for robust generalization
# - Loss = MSE(original, reconstructed) + 0.01 * PAPR(time_signal)
# - Result: Model saved as prnet_lambda_0.01.weights.h5
#
# USAGE:
#   python train_prnet.py                    # Train the model
#   python train_prnet.py --quick            # Quick test (reduced steps)
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 1: Import Libraries for Training
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Set up the training environment
#
# - os/sys/argparse: For file I/O and command-line arguments
# - time: For tracking training duration
# - numpy/tensorflow: For numerical computation and deep learning
# - prnet_model: The neural network we're training
# - ofdm_helpers: Data generation utilities
# ═══════════════════════════════════════════════════════════════════════════

import os, sys, argparse, time
import numpy as np
import tensorflow as tf
from prnet_model import PRNet
from ofdm_helpers import generate_qpsk_symbols, symbols_to_real

# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 2: Training Hyperparameters (from Paper Section IV)
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Define all training configuration values
#
# NETWORK ARCHITECTURE:
# - N = 64: Number of OFDM subcarriers (standard OFDM size)
# - HIDDEN_DIM = 2048: Neurons per hidden layer (large capacity for learning)
# - NUM_BLOCKS = 5: Number of hidden layers (depth for complex patterns)
#
# TRAINING SETTINGS:
# - BATCH_SIZE = 400: Process 400 OFDM symbols at once (paper setting)
# - TOTAL_STEPS = 100,000: Number of gradient updates (converges by ~100k)
# - LEARNING_RATE = 0.001: Step size for gradient descent (Adam optimizer)
#
# PAPR REDUCTION PARAMETERS:
# - LAMBDA = 0.01: Trade-off between BER and PAPR (paper's optimal value)
# - ETA_OPTIMAL = -15 dB: Optimal training noise level (found in Stage 1)
# - L = 4: Oversampling factor (required for accurate PAPR measurement)
#
# ═══════════════════════════════════════════════════════════════════════════

N              = 64          # subcarriers
HIDDEN_DIM     = 2048        # neurons per hidden layer
NUM_BLOCKS     = 5           # encoder/decoder sub-blocks
BATCH_SIZE     = 400         # "batch size … is set to 400"
TOTAL_STEPS    = 100_000     # reduced from 500k (model converges by ~100k)
LAMBDA         = 0.01        # "λ is set to 0.01"
ETA_OPTIMAL    = -15.0       # "PRNet trained at η = −15 dB"
LEARNING_RATE  = 1e-3
L              = 4           # oversampling factor

MODEL_DIR      = os.path.join(os.path.dirname(__file__), "models")
LOG_INTERVAL   = 2_000       # print progress every N steps


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 3: Training Data Generator
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Create infinite stream of random QPSK training data
#
# WHY RANDOM DATA?
# - PRNet should work on ANY QPSK data, not memorize specific patterns
# - Fresh random data every batch prevents overfitting
# - Infinite generator: never runs out of training examples
#
# FLOW:
# 1. Generate random bits
# 2. Modulate to QPSK symbols
# 3. Convert to real vectors (for network input)
# 4. Yield batch, repeat forever
#
# This is wrapped in tf.data.Dataset for efficient pipeline processing
# ═══════════════════════════════════════════════════════════════════════════

def data_generator(batch_size: int, N: int):
    # Yield random QPSK real-valued vectors forever.
    while True:
        # Generate random bits, modulate to QPSK, convert to real
        _, symbols = generate_qpsk_symbols(batch_size, N)
        yield symbols_to_real(symbols)


def make_dataset(batch_size: int, N: int) -> tf.data.Dataset:
    # Create a tf.data.Dataset from the generator.
    # Wrap Python generator in TensorFlow Dataset
    ds = tf.data.Dataset.from_generator(
        lambda: data_generator(batch_size, N),
        output_signature=tf.TensorSpec(shape=(batch_size, 2 * N),
                                       dtype=tf.float32)
    )
    # Prefetch batches for better GPU utilization
    return ds.prefetch(tf.data.AUTOTUNE)


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 4: Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Train PRNet model for specified number of gradient updates
#
# TRAINING PROCESS:
# 1. Compile model with Adam optimizer
# 2. For each step:
#    a. Get random batch of QPSK data
#    b. Forward pass: compute loss (MSE + λ*PAPR)
#    c. Backward pass: compute gradients
#    d. Update weights: optimizer.apply_gradients()
# 3. Log progress every 2000 steps
#
# LOGGED METRICS:
# - loss: Total loss (L1 + λ*L2)
# - mse: Reconstruction error (data fidelity)
# - papr: Mean PAPR in linear scale (efficiency measure)
# ═══════════════════════════════════════════════════════════════════════════

def train_model(model: PRNet,
                total_steps: int,
                batch_size: int,
                lr: float = 1e-3,
                tag: str = ""):
    # Train a PRNet model for *total_steps* gradient updates.
    # Setup optimizer (Adam with specified learning rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr))

    # Create infinite data stream
    ds = iter(make_dataset(batch_size, model.N))

    # Print training configuration
    print(f"\n{'='*60}")
    print(f"  Training {tag}")
    print(f"  Steps: {total_steps:,}   Batch: {batch_size}   "
          f"lambda={model.lambda_}   eta={model.eta_db} dB")
    print(f"{'='*60}\n")

    # Training loop
    t0 = time.time()
    for step in range(1, total_steps + 1):
        # Get next batch and perform one gradient update
        batch = next(ds)
        logs = model.train_step(batch)

        # Log progress periodically
        if step % LOG_INTERVAL == 0 or step == 1:
            elapsed = time.time() - t0
            eta_sec = elapsed / step * (total_steps - step)
            print(f"  [{tag}] step {step:>7,}/{total_steps:,}  "
                  f"loss={logs['loss']:.4f}  "
                  f"mse={logs['loss_signal']:.4f}  "
                  f"papr={logs['loss_papr']:.2f}  "
                  f"({elapsed:.0f}s elapsed, ~{eta_sec:.0f}s remaining)")

    # Training complete
    elapsed = time.time() - t0
    print(f"\n  [done] {tag} finished in {elapsed/60:.1f} min\n")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 5: Joint PAPR Reduction Training
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Train final model with both reconstruction AND PAPR objectives
#
# LOSS FUNCTION:
#   Total Loss = MSE(original, reconstructed) + λ * PAPR(time_signal)
#
# RESULT:
# - Encoder learns to modify symbols such that:
#   1. They can still be decoded correctly (low MSE)
#   2. Their IFFT has low PAPR (low PAPR term)
# - Saves model as "prnet_lambda_0.01.weights.h5"
# ═══════════════════════════════════════════════════════════════════════════

def train_joint_papr(lambda_, eta_db, total_steps, batch_size, lr):
    # Train PRNet with joint loss L = L1 + λ·L2.
    print("\n" + "#" * 60)
    print("  TRAINING — Joint objective (MSE + lambda*PAPR)")
    print("#" * 60)

    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Create model with both λ (PAPR penalty) and optimal η
    model = PRNet(N=N, L=L, num_blocks=NUM_BLOCKS,
                  hidden_dim=HIDDEN_DIM,
                  lambda_=lambda_,      # Includes PAPR penalty
                  eta_db=eta_db)        # Optimal corruption level

    # Train the model with joint loss
    tag = f"lambda={lambda_}"
    model = train_model(model, total_steps, batch_size, lr, tag=tag)

    # Save final trained weights
    path = os.path.join(MODEL_DIR, f"prnet_lambda_{lambda_}.weights.h5")
    model.save_weights(path)
    print(f"  -> Saved weights: {path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 6: Main Execution - Command-line Interface
# ═══════════════════════════════════════════════════════════════════════════
# PURPOSE: Parse arguments and execute training
#
# COMMAND-LINE OPTIONS:
# --quick: Fast training for testing (5K steps instead of 100K)
# --steps: Override default number of training steps
# --lambda: Override default PAPR loss weight (default 0.01)
# --eta: Override default corruption level (default -15)
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train PRNet")
    p.add_argument("--quick", action="store_true",
                   help="Quick test with reduced steps (5,000)")
    p.add_argument("--steps", type=int, default=None,
                   help="Override total training steps")
    p.add_argument("--lambda", dest="lambda_", type=float, default=LAMBDA)
    p.add_argument("--eta", type=float, default=ETA_OPTIMAL)
    return p.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    steps = args.steps or (5_000 if args.quick else TOTAL_STEPS)

    # Configure GPU memory growth (prevents TensorFlow from allocating all GPU memory)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Execute Training
    train_joint_papr(args.lambda_, args.eta, steps, BATCH_SIZE, LEARNING_RATE)

    print("\n[done] Training complete. Weights saved to:", MODEL_DIR)
