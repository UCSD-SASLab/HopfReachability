# Koopman-Hopf Reachability 

Here we pair Koopman theory with the Hopf solution for approximate reachability and robust control, associated with this paper https://arxiv.org/abs/2303.11590. This involves using a high-dimensional linear system to approximate the nonlinear differential game to solve approximate Backwards Reachable Sets (see [Slow Manifold Lifted Target Comparison](./SlowManifold_Target_Comparison.jl), [Duffing Game Convexity Comparison](./Duffing_BRS_Comparison.jl)) and the corresponding optimal controls (see [Glycolysis Control Comparison Sample](./Glycolysis_Control_Comparison.jl)).

<p align="center">
  <img src="./abstract_graphic.png" width="500">
</p>

We generate and load Koopman models from [PyKoopman](https://github.com/dynamicslab/pykoopman) and [AutoKoopman](https://github.com/EthanJamesLew/AutoKoopman) and use our [HopfReachability](https://github.com/UCSD-SASLab/HopfReachability) codebase to solve the Hopf solution. Note, this (nontrivially) requires defining a target in the lifted space that corresponds to the true target set and convexity of the problem has important implications (see paper).

Currently, the demos are limited to lifting with the identity function (Dynamic Mode Decomposition) and a Polynomial basis as they satisfy requirements for the Hopf problem (see paper). Check back for future work with more complicated lifting procedures involving NN's, RBF's, RFF's etc.

## Examples

Slow Manifold Lifted Target Comparison,
<p align="center">
  <img src="./SlowManifold_Contours.jpg">
</p>

Duffing Game Convexity Comparison,
<p align="center">
  <img src="./Duffing_BRS.jpg" width="700">
</p>

Glycolysis Control Comparison,
<p align="center">
  <img src="./Glycolysis_Control_Comparison_Sample.png">
</p>
