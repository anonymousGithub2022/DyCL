#!/bin/bash

if [ ! -d "./tmp" ];then
      mkdir -p ./tmp
fi

CodeDir="compile_model/src_code/"
TaskList="0 1 2"
for task in $TaskList
do
  echo "$task"
  python compile_onnx.py --eval_id="$task"
  echo "$CodeDir""$task".py
done

for task in $TaskList
do
  python compile_tvm.py --eval_id="$task"
done