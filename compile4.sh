#!/bin/bash
#
#rm -rf compile_model_bakeup
#mv compile_model/ compile_model_bakeup


if [ ! -d "./tmp" ];then
      mkdir -p ./tmp
fi

CodeDir="compile_model/src_code/"
TaskList="7 8"
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


