# PRNet - Rayleigh

Reference: M. Kim, W. Lee, and D.-H. Cho, "A Novel PAPR Reduction Scheme for
OFDM System Based on Deep Learning," IEEE Communications Letters, 2018.

This folder contains a TensorFlow/Keras reproduction of PRNet trained and
evaluated for a Rayleigh fading channel.

## Files

```text
ofdm_helpers.py        OFDM, QPSK, PAPR, CCDF, and channel helpers
prnet_model.py         PRNet encoder/decoder model
train_prnet.py         Training loop
eval_ber.py            BER evaluation over Rayleigh fading
eval_ccdf.py           Paper-compatible CCDF evaluation
plot_constellation.py  Encoder constellation plot
```

## Paper Parameters

| Parameter | Value |
| --- | --- |
| Subcarriers | 64 |
| Modulation | 4-QAM / QPSK |
| Channel | Rayleigh fading + AWGN |
| Batch size | 400 |
| Training batches in paper | 500,000 |
| Lambda | 0.01 |
| Corruption level eta | -15 dB |

This reproduction uses a `2*N=128` real/imaginary QPSK coordinate input,
whereas the original paper/code uses a `4*N=256` one-hot representation. The
network idea is the same: one whole OFDM symbol is mapped to 64 learned complex
subcarrier values.

## Implementation Differences That Matter

The authors' public TensorFlow 1 code differs from a clean mathematical
implementation in several ways:

- It uses a `4*N=256` one-hot input/output representation, while this project
  uses `2*N=128` real/imaginary QPSK coordinates.
- It effectively uses no oversampling in the released PRNet CCDF code.
- It trains with learning rate `1e-4`; this project uses `1e-3`.
- Its published PRNet CCDF metric appears to be the peak-amplitude proxy
  described below.

These differences explain why a sensible Rayleigh model can have a reasonable
BER curve but a CCDF curve shifted right of the paper's red curve when measured
with true standard PAPR.

## What Was Wrong With the CCDF

The standard physical PAPR definition is:

```text
PAPR_dB = 10*log10(max(|x|^2) / mean(|x|^2))
```

Using that definition, the saved Rayleigh PRNet weights give a PRNet point near
6 dB at `CCDF = 1e-3`. The paper's red PRNet curve is near 3 dB, so a direct
standard-PAPR evaluation does not visually match the paper.

The reason is in the authors' released TensorFlow code. Their PRNet CCDF
calculation stores:

```text
peak_power_symbol = max(abs(encoded_symbol_original))
CCDF input        = 10*log10(peak_power_symbol)
```

That is a peak-amplitude proxy, not the standard peak-to-average power ratio.
After RMS normalization, this proxy is approximately half of standard PAPR in
dB, which explains why the paper-compatible PRNet curve appears around 3 dB.

## How It Is Fixed Here

`eval_ccdf.py` now produces one PRNet CCDF curve only:

- Original OFDM is plotted as a black dashed baseline.
- PRNet is plotted as a solid red curve.
- The PRNet curve uses the paper-compatible peak-amplitude proxy, matching the
  authors' released code and the paper's Fig. 3 appearance.
- The default CCDF sample count is `25,000` OFDM symbols per curve.
- The default CCDF oversampling is `1`, matching the authors' released PRNet
  CCDF code.
- A fixed random seed is used by default so repeated readings are stable.

This is intentionally a paper-reproduction plot. The helper function
`compute_papr_db()` still implements the scientifically standard PAPR formula
for strict physical PAPR reporting, but `eval_ccdf.py` does not plot a second
standard-PRNet curve because that caused confusion and did not match the paper.

## BER Axis Fix

`eval_ber.py` treats the x-axis as `Eb/N0`, matching the paper. Internally, the
channel simulator still uses symbol energy, so QPSK adds `10*log10(2)` dB when
generating noise.

## Usage

```bash
python train_prnet.py
python train_prnet.py --quick

python eval_ber.py --no-show
python eval_ccdf.py --no-show
python plot_constellation.py
```

Useful CCDF options:

```bash
python eval_ccdf.py --num-symbols 25000 --no-show
python eval_ccdf.py --seed 1234 --no-show
```

Outputs are written to `figures/`. The main CCDF figure is:

```text
figures/ccdf_papr_rayleigh.jpg
```

## Interpretation Rule

Use `figures/ccdf_papr_rayleigh.jpg` when you want the paper-matching PRNet
CCDF curve. When reporting numeric CCDF values, state the metric explicitly:
the paper-compatible proxy is near 3 dB for the saved model at
`CCDF = 1e-3`, while the standard power-PAPR definition is near 6 dB.
