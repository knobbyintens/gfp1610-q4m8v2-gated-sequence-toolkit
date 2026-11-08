# Gated Sequence Toolkit

A general-purpose toolkit for building gated sparse sequence models. It ships
reference blocks for several published techniques, including SGRK-style
recurrent kernels, gated convolution blocks, sparse state mixers and
kernel-sparsity regularisers.

The design and evaluation protocol of this library follow
**Gated Sparsity in Recurrent Sequence Models: A Practitioner's Survey**
(preprint 2509.12345, Ibarra and Sen, 2025). The library is not tied to any
single paper implementation; the SGRK block is one of many supported layers.

## Contents

* `gst/layers.py` — layer zoo (SGRK-style block, gated convolution, state mixer)
* `gst/regularisers.py` — sparsity regularisers
* `benchmarks/` — shared benchmark harness
