# Graduation Project 2 — Machine Learning Techniques for PAPR Reduction in Multicarrier Communication Systems

**Ain Shams University · Faculty of Engineering · Electronics and Communications Engineering Department**  
**Academic Year 2025/2026 · Supervisor: Dr. Michael Ibrahim**

---

## Overview

This repository contains the simulation codebase, research materials, and documentation for **Part II** of our graduation project, which investigates the application of machine learning (ML) and deep learning (DL) methodologies to solve the Peak-to-Average Power Ratio (PAPR) problem in multicarrier communication systems.

---

## Project Team

| Name | Name |
|---|---|
| Nour-Eldin Mohamed Mostafa | Youssef Samir Sleem Ahmed |
| Muhammad Mahmoud Muhammad | Reem Nader Saeed |
| Basel Mohamed Ramadan | Ahmed Walid Hassan |
| Yosef Ahmed Mohamed | |

---

## The PAPR Problem in OFDM

Orthogonal Frequency Division Multiplexing (OFDM) is a cornerstone of modern wireless communications (e.g., 5G, Wi-Fi), offering high spectral efficiency and robustness against multi-path fading. However, a major inherent drawback of OFDM is its high Peak-to-Average Power Ratio (PAPR). 

Because an OFDM signal is the sum of many independent, phase-modulated subcarriers, there are instances where these subcarriers align constructively in phase. When this occurs, the instantaneous peak power of the transmitter signal spikes significantly above its average power. 

To transmit these peaks without distortion, the transmitter's High-Power Amplifier (HPA) must operate in its linear region, requiring a large power back-off. This leads to:
*   **Low Power Efficiency**: The transmitter consumes significantly more DC power, reducing the battery life of mobile terminals and increasing energy costs for base stations.
*   **Non-linear Distortions**: If the signal peaks exceed the amplifier's linear range, it clips the signal, introducing in-band distortion (degrading the Bit Error Rate) and out-of-band emissions (causing interference to adjacent channels).

---

## Machine Learning for PAPR Reduction

Classical PAPR reduction techniques (such as clipping & filtering, selective mapping, tone reservation, and partial transmit sequence) often suffer from significant limitations, including high computational complexity, spectral efficiency loss from transmitting side information, or high distortion.

Machine learning offers a paradigm shift in addressing these issues:
*   **End-to-End Optimization**: Deep learning architectures can treat the transmitter, channel, and receiver as a joint optimization problem, learning constellation mappings and pre-distortions that inherently exhibit low PAPR while maintaining low reconstruction error (BER).
*   **Low Execution Latency**: Once trained, forward passes through neural network models require simple matrix multiplications, which can be executed very rapidly on hardware, bypassing the exhaustive searches or iterative solvers required by classical methods.
*   **Eliminating Side Information**: ML models can learn to compress or pre-distort waveforms in a way that allows the receiver to decode the signal without requiring additional control or side information bits.

---

## Development Status

This repository serves as the development and collaboration space for our team. Various machine learning architectures, training datasets, and system simulations will be added as different components of the project are implemented and integrated.

---

*This project is submitted in partial fulfillment of the requirements for the degree of Bachelor of Science in Electronics and Communications Engineering, Ain Shams University.*
