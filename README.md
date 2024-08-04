# JointSQ
 Simple Implementation of the CVPR 2024 Paper "JointSQ: Joint Sparsification-Quantization for Distributed Learning"
 Paper：https://cvpr.thecvf.com/virtual/2024/poster/31122
# Overview
  In this paper, we propose adaptive Joint Sparsification Quantization (JointSQ) to address the suboptimal solution bottleneck for communication-efficient distributed learning. The conceptual framework of our approach can be seen in Figure 1. The key idea is to treat sparsification as 0-bit quantization thus the sparsification can indeed be unified with quantization fundamentally.
<p align="left">
<img src="JointSQ.png" width="700"><br/>
 Figure 1. Existing Co-compression methods and our JointSQ framework. Existing Co-compression methods typically apply sparsification and quantization step by step. Our framework considers sparsification as 0-bit quantization and thus the two-stage process is transformed into a unified learning framework.<br/><br/>
