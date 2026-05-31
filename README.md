# Multi-modal lifelog data fusion for improved human activity recognition: A hybrid approach

This repository contains the official code for a **hybrid multi-modal data fusion** approach to **human activity recognition (HAR)** from **lifelog** data. The method learns deep, nonlinear interactions between static and dynamic modalities and combines three kinds of interaction features: features from static data, latent representations learned by deep neural networks, and statistical features extracted from time series. It supports both **feature-level fusion** and **decision-level fusion**, and is evaluated on the ETRI-Lifelog data together with three widely-used public datasets: USC-HAD, UCI-HAR, and UCI-HAPT.

**Authors:** YongKyung Oh, Sungil Kim
**Venue:** Information Fusion (Elsevier), vol. 110, article 102464, October 2024
**DOI:** [10.1016/j.inffus.2024.102464](https://doi.org/10.1016/j.inffus.2024.102464)

**Keywords:** human activity recognition, multi-modal data fusion, lifelog, hybrid approach, feature-level fusion, decision-level fusion

## Overview

Human activity recognition is important in healthcare for personalized care and early intervention. Two central challenges are integrating heterogeneous multimodal data and extracting informative features. This work addresses both by fusing static and dynamic modalities through their deep, nonlinear interactions, and by exploring complementary fusion strategies. In addition to the existing ETRI-Lifelog data, the method is evaluated on three widely-used public datasets (USC-HAD, UCI-HAR, UCI-HAPT), showing that the proposed hybrid approach outperforms traditional methods.

## Repository structure

- `preprocess/` — data preparation: `preprocess.py`, `splits.py`, `sktime_format.py`.
- `early_fusion/` — early (feature-level) fusion baseline (`clf_ml.py`).
- `late_fusion/` — late (decision-level) fusion baseline (`clf_mm_late.py`).
- `joint_fusion/` — joint fusion baseline (`clf_mm_joint.py`).
- `hybrid_fusion/` — hybrid fusion (`clf_mm.py`).
- `proposed_fusion/` — proposed fusion model and variants (`clf_mm.py`, `clf_mme-fcn.py`, `clf_mme-lstm.py`, `clf_mme-resnet.py`).
- `lib.py` — shared utilities.

## Usage

1. Prepare and split the datasets using the scripts in `preprocess/`.
2. Run a fusion variant, e.g. the proposed model in `proposed_fusion/`, or compare against the baselines in `early_fusion/`, `late_fusion/`, `joint_fusion/`, and `hybrid_fusion/`.

## Citation

If you use this code, please cite:

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

```bibtex
@article{oh2024multimodal,
  title   = {Multi-modal lifelog data fusion for improved human activity recognition: A hybrid approach},
  author  = {Oh, YongKyung and Kim, Sungil},
  journal = {Information Fusion},
  volume  = {110},
  pages   = {102464},
  year    = {2024},
  doi     = {10.1016/j.inffus.2024.102464}
}
```

## License

Released under the [MIT License](LICENSE).
