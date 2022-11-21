#!/bin/bash

if [ ! -d "./tmp" ];then
      mkdir -p ./tmp
fi

CodeDir="compile_model/src_code/"
TaskList="0 1 2 3 4 5 6 7 8"


for task in $TaskList
do
  cp "$CodeDir""$task".py tmp/demo.py
  python compile_tvm.py --eval_id="$task"
  python evaluate_tvm.py --eval_id="$task"
  python evaluate_onnx.py --eval_id="$task"
done