# DyCL

**DyCL** is an automatic tool enabling existing *static* DL compilers to compile and deploy dynamic neural networks.
**DyCL** is a general-purpose tool that can release the power of existing *static* DL compilers in the context of dynamic neural networks without touching the complicated design of various DL compiler IR.


<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/DyCL/blob/main/fig/overview.png" width="2880" height="590" alt="Design Overview"/><br/>
</div>    

The above figure shows the design overview of DyCL, which includes five main steps: 
 1. Program Rewriting: rewrite the source Dynamic neural network program to make the contest of each sub-DNN's context explicit.
 2. HCFG Construction: generate the HCFG for compiling each sub-DNN.
 3. Graph Optimization: further optimize each sub-DNNs' computational graph to achieve less runtime overheads.
 4. Sub-DNN Compilation: automatic compile each sub-DNN through tracking each sub-DNN's input shape.
 6. Host-API Generation: generating the API function by rewriting the original dynamic neural network's AST and each DL compiler's sytanix.



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

* ./compileXXX.sh. - the bash script to reproduce our main results, where XXX is a integer number.

## How to Run

Since we have put our code into the **compileXXX.sh** script, so just `bash compileXXX.sh` to run our code.


## Limitation of Existing DL Compilers


<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/DyCL/blob/main/fig/error.png" width="740" height="300" alt="Design Overview"/><br/>
</div>    

The above figure shows the error distribution of applying DL compiler to compiler four different dynamic neural netwroks. The red is the error distribution of compiling ResNet, which contains no dynamic behavior. From the results, we observe that existing DL compiler fail to compile dynamic neural networks due to the large errors ($10^1$ ~ $10^4$).



