# PRNet - Rayleigh Evaluation

Reference: M. Kim, W. Lee, and D.-H. Cho, "A Novel PAPR Reduction Scheme for
OFDM System Based on Deep Learning," IEEE Communications Letters, 2018.

This folder contains a TensorFlow/Keras reproduction of PRNet evaluated for a 
Rayleigh fading channel. Note: Training instructions and code are handled 
separately via the Kaggle guide (`kaggle_training_guide.md`).

## Files

```text
ofdm_helpers.py        OFDM, QPSK, PAPR, CCDF, and channel helpers
prnet_model.py         PRNet encoder/decoder model
eval_ber.py            BER evaluation over Rayleigh fading
eval_ccdf.py           CCDF evaluation (Standard vs Paper-Compatible)
plot_constellation.py  Encoder constellation plot
```

## Paper Parameters

| Parameter | Value |
| --- | --- |
| Subcarriers | 64 |
| Modulation | 4-QAM / QPSK |
| Channel | Rayleigh fading + AWGN |
| Lambda | 0.01 |
| Corruption level eta | -15 dB |

This reproduction uses a `2*N=128` real/imaginary QPSK coordinate input,
whereas the original paper/code uses a `4*N=256` one-hot representation. The
network idea is the same: one whole OFDM symbol is mapped to 64 learned complex
subcarrier values.

## The CCDF Plot Discrepancy

The standard physical PAPR definition is:

```text
PAPR_dB = 10*log10(max(|x|^2) / mean(|x|^2))
```

Using that standard definition, the PRNet curve does not visually match the 
paper's red PRNet curve (which appears near 3 dB at `CCDF = 1e-3`). 

The reason lies in the authors' released TensorFlow code. Their PRNet CCDF
calculation uses a peak-amplitude proxy:

```text
peak_power_symbol = max(abs(encoded_symbol_original))
CCDF input        = 10*log10(peak_power_symbol)
```

After RMS normalization, this proxy is approximately half of standard PAPR in
dB, which explains why the paper-compatible PRNet curve appears around 3 dB.

## How It Is Addressed Here

`eval_ccdf.py` plots three curves to provide full transparency:

1. **Original OFDM (Standard PAPR):** Black dashed baseline.
2. **PRNet (Standard Physical PAPR):** Solid red curve. This is the true, scientifically standard PAPR of the PRNet waveform.
3. **PRNet (Authors' Proxy Metric):** Dashed red curve. This reproduces the peak-amplitude proxy matching the authors' released code and the paper's Fig. 3 appearance.

- The default CCDF sample count is `25,000` OFDM symbols per curve.
- The default CCDF oversampling is `1`, matching the authors' released PRNet
  CCDF code.
- A fixed random seed is used by default so repeated readings are stable.

## BER Axis Note

`eval_ber.py` treats the x-axis as `Eb/N0`, matching the paper. Internally, the
channel simulator still uses symbol energy, so QPSK adds `10*log10(2)` dB when
generating noise.

## Usage

*Ensure you have downloaded the trained model weights from Kaggle into the `models/` folder first.*

```bash
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

When reporting numeric CCDF values, state the metric explicitly. The plot clearly shows the divergence between the true physical PAPR of the network and the proxy metric published in the original paper.
