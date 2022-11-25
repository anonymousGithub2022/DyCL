# File Structure
* ./src - basic implementation of DyCL
  * ./src/onnx_rewrite.py - optimize the computational graph of a neural network that is stored in onnx format.
  * ./src/static_analysis.py - Basic static analysis module in DyCL. Implementing the basic functions for loop unrolling and constant propragration.
  * ./src/parse_cfg.py - The script used to extract the CFG.
  * ./src/dead_var_remove.py - optimize the code to remove the dead variable.

* ./compile_onnx.py - script to compile the dynamic neural networks into onnx format.
* ./compile_tvm.py - script to compile the dynamic neural networks into tvm format.
* ./evaluate_onnx.py - script to evaluate the compiled onnx dynamic neural networks.
* ./evaluate_tvm.py - script to evaluate the compiled onnx dynamic tvm networks.
* ./utils.py        - implementing some api function for our experiments.



