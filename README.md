# Flexible Communication for Optimal Distributed Learning over Unpredictable Networks

**Code for AR-Topk compression published in IEEE International Conference on Big Data (BigData), 2023, Sorrento, Italy.**

_Gradient compression alleviates expensive communication in distributed deep learning by sending fewer values and its corresponding indices, typically via Allgather (AG). 
Training with high compression ratio (CR) achieves high accuracy like DenseSGD, but has lower parallel scaling due to high communication cost (i.e., parallel efficiency). 
Using lower CRs improves parallel efficiency by lowering synchronization cost, but degrades model accuracy as well (statistical efficiency). 
Further, speedup attained with different models and CRs also varies with network latency, effective bandwidth and collective op used for aggregation. 
In many cases, collectives like Allreduce (AR) have lower cost than AG to exchange the same amount of data. 
In this work, we propose an AR-compatible Topk compressor that is bandwidth-optimal and thus performs better than AG in certain network configurations. 
We develop a flexible communication strategy that switches between AG and AR based on which collective is optimal in the current settings, and model the pareto-relationship between parallel and statistical efficiency as a multi-objective optimization (MOO) problem to dynamically adjust CR and accelerate training while still converging to high accuracy._

**ACCESS LINKS**

- [Link1](https://ieeexplore.ieee.org/document/10386724/)
- [Link2](https://sahiltyagi4.github.io/files/adopt.pdf)

**CITATION**
- **_Bibtex_**: @article{Tyagi2023FlexibleCF,
  title={Flexible Communication for Optimal Distributed Learning over Unpredictable Networks},
  author={Sahil Tyagi and Martin Swany},
  journal={2023 IEEE International Conference on Big Data (BigData)},
  year={2023},
  pages={925-935}}