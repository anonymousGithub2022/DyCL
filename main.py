import copy
import time
from tqdm import tqdm
import numpy as np
from src import to_graph, gen_cfg, DNNClassCFG, get_cfg, MyConfigClass
from src import AdNNClassCFG
import inspect
import ast
import os
import torch
from collections import namedtuple

from utils import *

print(os.getcwd())
if not os.path.isdir('tmp'):
    os.mkdir('tmp')

if __name__ == '__main__':
    basemodel, src, compile_func, example_x, test_loader = load_model(6)
    basemodel = basemodel.eval()
    # compile_func = 'test'
    analysis = AdNNClassCFG('demo', src, basemodel, compile_func)

    # basemodel.forward = eval('basemodel.%s' % compile_func)
    # torch.jit.trace(basemodel, example_x['x'])
    # print('test example')

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

    ##########################################
    # above codes seperate the adNN into a set of DNNs and a general API
    ###########################################

    from tmp.demo import MyDNN
    model = MyDNN(basemodel)
    model = model.eval().to('cpu')
    entry_id = 1
    next_id_list = [entry_id]
    is_visited = {}
    for k in analysis.node_ids:
        is_visited[k] = False

    new_model_dict = {}
    out_res = {}
    with torch.no_grad():
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
            #try:
            new_model = torch.jit.trace(model, example_inputs=input_x)
            # except:
            #     new_model = None
            #     print(model_name, 'error')

            new_model_dict[model_name] = new_model

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
    for key in new_model_dict:
        model_instance = getattr(model, key)
        ori_model_dict[key] = model_instance

    compile_time, ori_time = [], []
    input_shape = example_x['x'].shape
    for x in test_loader:
        example_x = dict()
        example_x['x'] = torch.clone(x)

        pred_y = predictAPI(example_x, ori_model_dict, self)
        print(pred_y)
        print()

        t1 = time.time()
        basemodel = basemodel.eval()
        basemodel.forward = eval('basemodel.%s' % compile_func)
        ori_y = basemodel(x)
        print(ori_y)
        print()
        t2 = time.time()
        ori_time.append(t2 - t1)

        example_x['x'] = torch.clone(x)
        t1 = time.time()
        compile_pred_y = predictAPI(example_x, new_model_dict, self)
        print(compile_pred_y)
        t2 = time.time()
        compile_time.append(t2 - t1)

        print('-------------------------')
print()
