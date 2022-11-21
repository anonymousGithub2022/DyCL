import os
import torch
import numpy as np
from csv import writer

from utils import RES_DIR

f = open('final_res.csv', 'w')
csv_writer = writer(f)
csv_writer.writerow([
    'task_id', 'device id', 'cl', 'graph optimize',
    'analysis_time', 'onnx compile time', 'graph opt time', 'tvm time',
    'max_error', 'is_success',
    'ori_exe_time_avg', 'ori_exe_time_std',
    'compile_exe_time_avg', 'compile_exe_time_std',
])

prefix_list = ['ON_OPT_', 'OPT_']
tvm_compile_time_key_list = ['NO_OPT_tvm_compile_time', 'OPT_tvm_compile_time']
for optimize in [0, 1]:
    prefix = prefix_list[optimize]
    p = tvm_compile_time_key_list[optimize]
    for device in [0]:
        for cl in ['tvm', 'onnx']:
            for task_id in range(9):
                try:
                    sub_file_name = str(task_id) + '_' + str(device) + '_' + str(optimize) + '_' + cl + '.tar'
                    file_name = os.path.join(RES_DIR, sub_file_name)
                    trace_res = torch.load(file_name)
                    print(sub_file_name)

                    res = [
                        task_id, device, cl, optimize,
                        trace_res['analysis time'], trace_res['onnx compile time'],
                        trace_res['graph optimization time'], trace_res[p],
                        trace_res[prefix + 'max error'], trace_res[prefix + 'is success'],
                        np.mean(trace_res[prefix + 'ori exe runtime']), np.std(trace_res[prefix + 'ori exe runtime']),
                        np.mean(trace_res[prefix + 'compile exe runtime']), np.std(trace_res[prefix + 'compile exe runtime']),

                    ]
                except:
                    res = [
                        task_id, device, cl, optimize,
                        -1, -1,
                        -1, -1,
                        -1, -1,
                        -1, -1,
                        -1,
                        -1,

                    ]
                csv_writer.writerow(res)


f.close()