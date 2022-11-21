import os
TensorRTPATH = "/disk/CM/DLCompiler/TensorRT-8.4.0.6"
if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = TensorRTPATH + "/lib"
else:
    os.environ['LD_LIBRARY_PATH'] = TensorRTPATH + "/lib:" + os.environ['LD_LIBRARY_PATH']
import torch
import argparse
from pathlib import Path
import numpy as np

from utils import META_DIR, test_ONNX_correctness
from utils import load_model, load_onnx_runtime
from utils import OPTIMIZE_ONNX, ONNX_DIR, RES_DIR

from tmp import TRT_API_OPT
from tmp import TRT_API_No_OPT


def main(task_id, is_optimize):
    pred_api_list = [TRT_API_No_OPT, TRT_API_OPT]
    pred_api = pred_api_list[is_optimize]

    if is_optimize:
        tvm_lib_dir = OPTIMIZE_ONNX
        prefix = 'OPT_'
        constant_key = 'opt_constant_dict'
    else:
        tvm_lib_dir = ONNX_DIR
        prefix = 'ON_OPT_'
        constant_key = 'no_opt_constant_dict'

    basemodel, src, compile_func, example_x, test_loader = load_model(task_id)

    onnx_model_dict = load_trt_runtime(str(tvm_lib_dir), task_id)

    meta_path = os.path.join(META_DIR, str(task_id) + '_' + str(device_id) + '_tvm_meta.tar')
    ori_state_dict = torch.load(meta_path)
    constant_dict = ori_state_dict[constant_key]

    for k in constant_dict:
        constant_dict[k] = constant_dict[k].detach().numpy()

    is_success, max_error, trace_err_his, onnx_runtime, ori_runtime = \
        test_ONNX_correctness(
            basemodel, pred_api, onnx_model_dict,
            test_loader, compile_func, constant_dict, device
        )

    meta_path = os.path.join(META_DIR, str(task_id) + '_' + str(device_id) + '_tvm_meta.tar')
    ori_state_dict = torch.load(meta_path)
    state_dict = {
        'task id:': task_id,
        'device': device_id,
        prefix + 'is success': is_success,
        prefix + 'max error': max_error,
        prefix + 'compile exe runtime': onnx_runtime,
        prefix + 'ori exe runtime': ori_runtime,
    }
    print(
        'task id:', task_id, '\n',
        'device', device_id, '\n',
        prefix + 'is success', is_success, '\n',
        prefix + 'max error', max_error, '\n',
        prefix + 'compile exe runtime', np.mean(onnx_runtime), '\n',
        prefix + 'ori exe runtime', np.mean(ori_runtime), '\n',
    )
    state_dict['trace_error_his'] = trace_err_his
    ori_state_dict.update(state_dict)

    res_path = os.path.join(RES_DIR, str(task_id) + '_' + str(device_id) + '_' + str(is_optimize) + '_onnx.tar')

    torch.save(ori_state_dict, res_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluate onnx runtime model")
    parser.add_argument("--eval_id", default=4, type=int, help="configuration file")
    args = parser.parse_args()

    for optimize in [0, 1]:
        main(args.eval_id, optimize)
