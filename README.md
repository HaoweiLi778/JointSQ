# JointSQ
 Simple Implementation of the CVPR 2024 Paper "JointSQ: Joint Sparsification-Quantization for Distributed Learning"
 Paper：https://cvpr.thecvf.com/virtual/2024/poster/31122
# Overview
  We propose Joint Sparsification Quantization (JointSQ), to address suboptimal solutions in communication-efficient distributed learning. Our approach unifies sparsification and quantization by treating sparsification as 0-bit quantization. JointSQ involves mixed-bit precision quantization for end-to-end compression. To adaptively assign bit-widths, we introduce a specially designed Multiple-Choice Knapsack Problem (MCKP) per layer with minimal computational cost.
<p align="center">
<img src="JointSQ.png" width="700"><br/>
 Figure 1. Existing Co-compression methods and our JointSQ framework.  <br/><br/>

 
