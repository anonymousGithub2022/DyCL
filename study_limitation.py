import copy
import time
from tqdm import tqdm
import numpy as np
from src import to_graph, gen_cfg, DNNClassCFG, get_cfg, MyConfigClass
from src import AdNNClassCFG
import argparse

import os

from utils import *

if not os.path.isdir('res'):
    os.makedirs('res')

def main(eval_id):
    task_name = 'demo'
    basemodel, src, compile_func, example_x, test_loader = load_model(eval_id)
    basemodel = basemodel.eval().to('cpu')
    analysis = AdNNClassCFG(task_name, src, basemodel, compile_func)

    basemodel.forward = eval('basemodel.%s' % compile_func)
    nvtive_compile_model = torch.jit.trace(basemodel, example_x['x'])

    input_var = ['x']
    io_var_list = analysis.get_var_list(input_var)
    io_var_list = analysis.refine_var_list(io_var_list)

    for k in analysis.node_ids:
        print('child:', k, io_var_list[k][0])
        child_list = analysis.parent_edges[k]
        for c_id in child_list:
            # res.append(io_var_list[c_id][0])
            print('parent', c_id, io_var_list[c_id][1])
        print('-----------------')
    model_list = analysis.prepare_model_list(io_var_list)

    # ast_body = copy.deepcopy()
    analysis.synthesis_python(io_var_list)
    print('---------------------------')

    time.sleep(3)
    ##########################################
    # above codes seperate the adNN into a set of DNNs and a general API
    ###########################################
    from tmp.demo import MyDNN
    # from tmp.demo import MyDNN

    model = MyDNN(basemodel)
    model = model.eval().to('cpu')
    entry_id = 1
    next_id_list = [entry_id]
    is_visited = {}
    for k in analysis.node_ids:
        is_visited[k] = False

    compile_model_dict = {}
    out_res = {}

    while len(next_id_list):
        current_id = next_id_list[0]
        is_visited[current_id] = True
        next_id_list = next_id_list[1:]
        child_is_list = analysis.child_edges[current_id]
        for child_id in child_is_list:
            if not is_visited[child_id] and child_id not in next_id_list:
                next_id_list.append(child_id)

        if current_id not in analysis.id2func:
            parent_list = analysis.parent_edges[current_id]
            out_res[current_id] = out_res[parent_list[0]]
            continue
        model_name = analysis.id2func[current_id]
        model.forward = eval('model.%s' % model_name)

        parent_list = analysis.parent_edges[current_id]
        if not parent_list:
            input_x = example_x
        else:
            input_x = out_res[parent_list[0]]

        print('begin compile model %s' % model_name)
        # new_model = torch.jit.script(model, example_inputs=input_x)
        # try:
        new_model = torch.jit.trace(model, example_inputs=input_x)
        # except:
        #     new_model = None
        #     print(model_name, 'error')

        compile_model_dict[model_name] = new_model

        out_x = model(input_dict=input_x)

        out_val_names = io_var_list[current_id][1]
        store_out_x = {}
        for k, v in zip(out_val_names, out_x):
            store_out_x[k] = v
        out_res[current_id] = store_out_x

        print(current_id, model_name, 'successful')

    ##########################################
    # above codes compile each DNN
    ###########################################

    #############################
    ##  test the correctness
    ############################

    from tmp.demo import predictAPI

    self = MyConfigClass(basemodel)

    ori_model_dict = {}
    for key in compile_model_dict:
        model_instance = getattr(model, key)
        ori_model_dict[key] = model_instance

    err_list = []
    for i, x in enumerate(test_loader):
        if i >= 1000:
            break
        example_x['x'] = torch.clone(x)
        compile_pred_y = predictAPI(example_x, compile_model_dict, self)

        ori_x = torch.clone(x)
        nvtive_pred_y = nvtive_compile_model(ori_x)
        if nvtive_pred_y is not torch.tensor:
            compile_pred_y = compile_pred_y[0]
            nvtive_pred_y = nvtive_pred_y[0]
        if compile_pred_y.size() == nvtive_pred_y.size():
            max_error = (compile_pred_y - nvtive_pred_y).max()
        else:
            max_error = (compile_pred_y.to(torch.float).mean() - nvtive_pred_y.to(torch.float).mean()).max()
        err_list.append(float(max_error))
    err_list = np.array(err_list).reshape([-1, 1])
    np.savetxt('res/study_error_%d.csv'%eval_id, err_list, delimiter=',')
    return err_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="compile model id")
    parser.add_argument("--eval_id", default=0, help="configuration file")
    args = parser.parse_args()

    err_list = main(int(args.eval_id))