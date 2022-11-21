import os
import argparse
import numpy as np
import torch.jit

from utils import *


def load_trace_model(model_dir):
    model_dict = {}
    for name in os.listdir(model_dir):
        model_dict[name] = torch.jit.load(os.path.join(model_dir, name))
    return model_dict


def main(task_id):
    basemodel, src, compile_func, example_x, test_loader = load_model(task_id)
    basemodel = basemodel.eval()

    optimize_model_dict = load_trace_model(os.path.join(OPT_TRACE, str(task_id)))
    trace_model_dict = load_trace_model(os.path.join(TRACE_DIR, str(task_id)))

    meta_path = os.path.join(META_DIR, str(task_id) + '_meta.tar')
    ori_state_dict = torch.load(meta_path)
    constant_dict = ori_state_dict['constant_dict']

    from tmp.demo import predictAPI, ONNX_API, TVM_API, TVM_API_Binary
    _, _, _, optimize_compile_time, optimize_ori_time = \
        test_JIT_correctness(
            basemodel, predictAPI, optimize_model_dict, test_loader, compile_func, constant_dict
        )

    is_success, max_error, trace_err_his, compile_time, ori_time = \
        test_JIT_correctness(
            basemodel, predictAPI, trace_model_dict, test_loader, compile_func, constant_dict
        )

    layers = count_model_layers(basemodel)
    print(
        'task id:', task_id, '\n',
        'number of layers', len(layers), '\n',
        'number of sub model', len(trace_model_dict), '\n',
        'is success', is_success, '\n',
        'max error', max_error, '\n',
        'torchtrace_runtime', np.mean(compile_time), '\n',
        'ori_runtime', np.mean(ori_time), '\n',
        'optimize_torchtrace_time', np.mean(optimize_compile_time), '\n',
        'optimize_ori_time', np.mean(optimize_ori_time),
    )


    state_dict = {
        'torchtrace_runtime': compile_time,
        'ori_runtime': ori_time,
        'optimize_compile_time': optimize_compile_time,
        'optimize_ori_time': optimize_ori_time,
        'layer_num': len(layers),
        'max_error': max_error,
        'trace_error_his': trace_err_his,
        'predictAPI': inspect.getsource(predictAPI),
        'ONNX_API': inspect.getsource(ONNX_API),
        'TVM_API': inspect.getsource(TVM_API),
        'TVM_API_Binary': inspect.getsource(TVM_API_Binary),
    }
    state_dict.update(ori_state_dict)
    torch.save(state_dict, meta_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="compile model id")
    parser.add_argument("--eval_id", default=4, help="configuration file")
    args = parser.parse_args()

    main(args.eval_id)
    # main(int(args.eval_id))
    exit(0)
