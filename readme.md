


# DyCL 


<p align="center">
 <a href="https://github.com/anonymousGithub2022/main/LICENSE"><img src="https://img.shields.io/github/license/anonymousGithub2022/DyCL"></a>
 <a href="https://github.com/anonymousGithub2022/main/LICENSE"><img src="https://img.shields.io/pypi/pyversions/tvm"></a>
 <a href="https://github.com/anonymousGithub2022/main/LICENSE"><img src="https://img.shields.io/github/languages/code-size/anonymousGithub2022/DyCL"></a>
</p>


<div  align="center">    
 <img src="https://github.com/anonymousGithub2022/DyCL/blob/main/fig/overview.png" width="680" height="290" alt="Design Overview"/><br/>
</div>    

**DyCL** is an automatic tool enabling existing *static* DL compilers to compile and deploy dynamic neural networks.
**DyCL** is a general-purpose tool that can release the power of existing *static* DL compilers in the context of dynamic neural networks without touching the complicated design of various DL compiler IR.

The above figure shows the design overview of DyCL, which includes five main steps: 
 1. Program Rewriting: rewrite the source Dynamic neural network program to make the contest of each sub-DNN's context explicit.
 2. HCFG Construction: generate the HCFG for compiling each sub-DNN.
 3. Graph Optimization: further optimize each sub-DNNs' computational graph to achieve less runtime overheads.
 4. Sub-DNN Compilation: automatic compile each sub-DNN through tracking each sub-DNN's input shape.
 6. Host-API Generation: generating the API function by rewriting the original dynamic neural network's AST and each DL compiler's sytanix.

# Dynamic Neural Networks in our Evaluation

| **ID.** | **Base DNN**     | **Application**      | **# of Layers** | **# of Sub-DNN** | **Model Size(MB)** | **Paper Link**                               | **GitHub**                                                 |
|---------|------------------|----------------------|-----------------|------------------|--------------------|----------------------------------------------|-------------------------------------------------------------|
| **1**   | MobileNet        | Image Classification | 142             | 108              | 354                | [[Paper 1]](http://shallowdeep.network/)                  | [Github 1](https://github.com/yigitcankaya/Shallow-Deep-Networks)       |
| **2**   | VGG19            | Image Classification | 115             | 42               | 430                | [[Paper 2]](http://shallowdeep.network/)                  | [Github 2](https://github.com/yigitcankaya/Shallow-Deep-Networks)       |
| **3**   | ResNet50         | Image Classification | 308             | 81               | 21                 | [[Paper 3]](http://shallowdeep.network/)                  | [Github 3](https://github.com/yigitcankaya/Shallow-Deep-Networks)       |
| **4**   | WideResNet       | Image Classification | 201             | 20               | 99                 | [[Paper 4]](http://shallowdeep.network/)                  | [Github 4](https://github.com/yigitcankaya/Shallow-Deep-Networks)       |
| **5**   | ResNet38 + RNN   | Image Classification | 179             | 108              | 2.3                | [[Paper 5]](https://arxiv.org/abs/1711.09485)             | [Github 5](https://github.com/ucbdrive/skipnet)                         |
| **6**   | ResNet38 + FC    | Image Classification | 336             | 111              | 4.6                | [[Paper 6]](https://arxiv.org/abs/1711.09485)             | [Github 6](https://github.com/ucbdrive/skipnet)                         |
| **7**   | ResxNext + LSTM  | Image Caption        | 139             | 200              | 68                 | [[Paper 7]](https://proceedings.mlr.press/v37/xuc15.html) | [Github 7](https://github.com/parksunwoo/show_attend_and_tell_pytorch)  |
| **8**   | GoogLeNet + LSTM | Image Caption        | 303             | 200              | 249                | [[Paper 8]](https://proceedings.mlr.press/v37/xuc15.html) | [Github 8](https://github.com/parksunwoo/show_attend_and_tell_pytorch)  |
| **9**   | FlatResNet32     | Image Classification | 320             | 42               | 7.2                | [[Paper 9]](https://arxiv.org/abs/1711.08393)             | [Github 9](https://github.com/Tushar-N/blockdrop)                       |


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

The above figure shows the error distribution of applying DL compiler to compiler four different dynamic neural netwroks. The red is the error distribution of compiling ResNet, which contains no dynamic behavior. From the results, we observe that existing DL compiler fail to compile dynamic neural networks due to the large errors （ $10^1$ ~ $10^4$ ).


## More Ablation Study Results (S)

| **ID.**  | **TVM** | **TVM** | **TVM**    | **ONNX** | **ONNX** | **ONNX**        |
|----------|---------|----------|-------------|----------|----------|--------------|
| **null** | V_{no}  | V        | Accelerate  | V_{no}   | V        | Accelerate   |
| **1**   | 2.110   | 2.101    | 0.449       | 1.137    | 1.117    | 1.717        |
| **2**   | 1.129   | 1.032    | 8.553       | 1.525    | 1.401    | 8.149        |
| **3**   | 0.102   | 0.103    | -0.599      | 0.167    | 0.173    | -3.193       |
| **4**   | 1.200   | 0.937    | 21.878      | 1.500    | 1.304    | 13.034       |
| **5**   | 0.072   | 0.071    | 1.641       | 0.045    | 0.045    | 0.848        |
| **6**   | 0.067   | 0.065    | 2.324       | 0.036    | 0.035    | 3.098        |
| **7**   | 0.516   | 0.505    | 2.127       | 0.397    | 0.396    | 0.208        |
| **8**   | 0.800   | 0.793    | 0.968       | 0.760    | 0.755    | 0.608        |
| **9**   | 0.096   | 0.096    | 0.255       | 0.076    | 0.074    | 2.331        |
| **Avg**  | 0.677   | 0.634    | 4.177       | 0.627    | 0.589    | 2.978        |

The above table shows the ablation study results on Nvidia TX2, where **V_{no}** represents the overehad (s) of the compiled DyNN without our proposed graph optimization module and **V** is the overehad (s) of the compiled DyNN from our approoach.
Similar to the results on Nvidia AGX, our proposed graph optimization module can further the compiled DyNN.

