import os
import sys
import ast
import astor
import unittest
from src.myscalpel import MyMNode
from ast import NodeTransformer

from scalpel.cfg.builder import Block

code_str = """
a = a + b
if a == 0:
    if b == 0:
        b = b + 1
s = s - 1 
"""

#
# class RewriteName(NodeTransformer):
#     def visit_Name(self, node):
#         return copy_location(Subscript(
#             value=Name(id='data', ctx=Load()),
#             slice=Index(value=Constant(value=node.id)),
#             ctx=node.ctx
#         ), node)

def get_iteration(ast_statement):
    return 2



def rewrite_statement(ori_s, index_name, index_val):
    constant = ast.Constant(index_val)

def unroll_loops(body, iter_num, index_name):
    new_ast_list = []
    for i in range(iter_num):
        new_code = index_name + '=' + str(i)
        assign_val_statement = ast.parse(new_code)
        new_ast_list.append(assign_val_statement)
        new_ast_list.extend(body)
        # for ori_s in body:
        #     # new_s = rewrite_statement(ori_s, index_name, i)
        #     new_ast_list.append(ori_s)
    return new_ast_list

def ast_rewrite(ori_ast):
    ast_body = ori_ast.body
    new_body = []
    for ast_statement in ast_body:
        if type(ast_statement) in [ast.For, ast.While]:
            index_name = ast_statement.target.id
            iteration_num = get_iteration(ast_statement)
            loop_body = ast_statement.body
            new_ast_statement = unroll_loops(loop_body, iteration_num, index_name)
            new_body.extend(new_ast_statement)
        else:
            new_body.append(ast_statement)

    return new_body

def main():
    mnode = MyMNode("local")
    mnode.source = code_str
    mnode.gen_ast()
    new_ast = mnode.ast_rewrite(mnode.ast)
    mnode.ast.body = new_ast
    cfg = mnode.gen_cfg(mnode.ast)
    print()
    # m_ssa = SSA()
    # ssa_results, const_dict = m_ssa.compute_SSA(cfg)
    # for block_id, stmt_res in ssa_results.items():
    #     print("These are the results for block ".format(block_id))
    #     print(stmt_res)
    # for name, value in const_dict.items():
    #     print(name, value)
    # print(ssa_results)


if __name__ == '__main__':
    main()

    # g = DNNClassCFG(src)
    # g.getFuncList()