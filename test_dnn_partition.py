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
    basemodel, src, compile_func, example_x, test_loader = load_model(0)
    basemodel = basemodel.eval()
    # compile_func = 'test'
    analysis = AdNNClassCFG('demo', src, basemodel, compile_func)

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

    from tmp.demo import MyDNN
    model = MyDNN(basemodel)
    model = model.eval().to('cpu')
    entry_id = 1
    next_id_list = [entry_id]
    is_visited = {}
    for k in analysis.node_ids:
        is_visited[k] = False

    from tmp.demo import predictAPI

    self = MyConfigClass(basemodel)

    # self.word_map = basemodel.word_map
    # word_map = {}
    # for k in self.word_map:
    #     word_map[self.word_map[k]] = k
    ori_model_dict = {}
    for key in analysis.id2func:
        model_instance = getattr(model, analysis.id2func[key])
        ori_model_dict[analysis.id2func[key]] = model_instance

    compile_time, ori_time = [], []
    input_shape = example_x['x'].shape

    for x in test_loader:
        example_x = dict()
        example_x['x'] = torch.clone(x)

        pred_y = predictAPI(example_x, ori_model_dict, self)
        print(pred_y)
        # print([word_map[int(tk)] for tk in pred_y[0]])

        t1 = time.time()
        basemodel = basemodel.eval()
        basemodel.forward = eval('basemodel.%s' % compile_func)
        ori_y = basemodel(x)
        print(ori_y)
        # print([word_map[int(tk)] for tk in ori_y[0]])
        t2 = time.time()
        ori_time.append(t2 - t1)

        print('-------------------------')
        # error = (pred_y[0] == ori_y[0])
        # print(pred_y[1],  ori_y[1])
        # if not error.all():
        #     print('max error', (pred_y[0] - ori_y[0]).max())
print()
