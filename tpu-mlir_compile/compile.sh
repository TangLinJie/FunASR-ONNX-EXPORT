#!/bin/bash

work_dir="../models"
if [ ! -d "$work_dir/onnx" ]; then
    echo "[Err] "$work_dir/onnx" directory not found, please export onnx model first"
    exit 1
fi

if [ ! $1 ]; then
    target=bm1684x
    target_dir=BM1684X
else
    target=${1,,}
    target_dir=${target^^}
    if test $target = "bm1684"
    then
        echo "do not support bm1684"
        exit
    fi
fi

bmodel_dir=../models/$target_dir
if [ ! -d "$bmodel_dir" ]; then
    mkdir "$bmodel_dir"
    echo "[Cmd] mkdir $bmodel_dir"
fi


pushd "$work_dir"

function gen_static_fp16bmodel()
{
    model_transform.py \
        --model_name $1 \
        --model_def ${2} \
        --input_shapes $3 \
        --mlir transformed.mlir
    
    model_deploy.py \
        --mlir transformed.mlir \
        --quantize F16 \
        --chip $target \
        --model $4
    mv $4 $bmodel_dir/
}

function gen_static_fp32bmodel()
{
    model_transform.py \
        --model_name $1 \
        --model_def ${2} \
        --input_shapes $3 \
        --mlir transformed.mlir
    
    model_deploy.py \
        --mlir transformed.mlir \
        --quantize F32 \
        --chip $target \
        --model $4
    mv $4 $bmodel_dir/
}

function gen_dynamic_fp16bmodel()
{
    model_transform.py \
        --model_name $1 \
        --model_def ${2} \
        --input_shapes $3 \
        --dynamic \
        --mlir transformed.mlir
    
    model_deploy.py \
        --mlir transformed.mlir \
        --quantize F16 \
        --chip $target \
        --dynamic \
        --model $4
    mv $4 $bmodel_dir/
}

function gen_dynamic_fp32bmodel()
{
    model_transform.py \
        --model_name $1 \
        --model_def ${2} \
        --input_shapes $3 \
        --dynamic \
        --mlir transformed.mlir
    
    model_deploy.py \
        --mlir transformed.mlir \
        --quantize F32 \
        --chip $target \
        --dynamic \
        --model $4
    mv $4 $bmodel_dir/
}

function gen_dynamic_encoder_fp16bmodel()
{
    model_transform.py \
        --model_name $1 \
        --model_def ${2} \
        --input_shapes $3 \
        --dynamic \
        --mlir transformed.mlir
    
    model_deploy.py \
        --mlir transformed.mlir \
        --quantize F16 \
        --chip $target \
        --dynamic \
        --quantize_table ../tpu-mlir_compile/encoder_qtable \
        --disable_layer_group \
        --model $4
    mv $4 $bmodel_dir/
}

function gen_dynamic_encoder_fp32bmodel()
{
    model_transform.py \
        --model_name $1 \
        --model_def ${2} \
        --input_shapes $3 \
        --dynamic \
        --mlir transformed.mlir
    
    model_deploy.py \
        --mlir transformed.mlir \
        --quantize F32 \
        --chip $target \
        --dynamic \
        --disable_layer_group \
        --model $4
    mv $4 $bmodel_dir/
}

# static
# encoder_model_name=encoder
# encoder_onnx_file=onnx/encoder.onnx
# encoder_input_shapes=[[1,1000,560],[1,1,1000]]
# encoder_bmodel_file=encoder_f16.bmodel 
# gen_static_fp16bmodel $encoder_model_name $encoder_onnx_file $encoder_input_shapes $encoder_bmodel_file
# encoder_model_name=encoder
# encoder_onnx_file=onnx/encoder.onnx
# encoder_input_shapes=[[1,1000,560],[1,1,1000]]
# encoder_bmodel_file=encoder_f32.bmodel 
# gen_static_fp32bmodel $encoder_model_name $encoder_onnx_file $encoder_input_shapes $encoder_bmodel_file

# dynamic
encoder_model_name=encoder
encoder_onnx_file=onnx/encoder.onnx
encoder_input_shapes=[[1,1000,560]]
encoder_bmodel_file=encoder_f16.bmodel 
gen_dynamic_encoder_fp16bmodel $encoder_model_name $encoder_onnx_file $encoder_input_shapes $encoder_bmodel_file

encoder_model_name=encoder
encoder_onnx_file=onnx/encoder.onnx
encoder_input_shapes=[[1,1000,560]]
encoder_bmodel_file=encoder_f32.bmodel 
gen_dynamic_encoder_fp32bmodel $encoder_model_name $encoder_onnx_file $encoder_input_shapes $encoder_bmodel_file

# static
# calc_predictor_model_name=calc_predictor
# calc_predictor_onnx_file=onnx/calc_predictor.onnx
# calc_predictor_input_shapes=[[1,1000,512],[1,1,1000]]
# calc_predictor_bmodel_file=calc_predictor_f16.bmodel 
# gen_static_fp16bmodel $calc_predictor_model_name $calc_predictor_onnx_file $calc_predictor_input_shapes $calc_predictor_bmodel_file
# calc_predictor_model_name=calc_predictor
# calc_predictor_onnx_file=onnx/calc_predictor.onnx
# calc_predictor_input_shapes=[[1,1000,512],[1,1,1000]]
# calc_predictor_bmodel_file=calc_predictor_f32.bmodel 
# gen_static_fp32bmodel $calc_predictor_model_name $calc_predictor_onnx_file $calc_predictor_input_shapes $calc_predictor_bmodel_file

# dynamic
calc_predictor_model_name=calc_predictor
calc_predictor_onnx_file=onnx/calc_predictor.onnx
calc_predictor_input_shapes=[[1,1000,512]]
calc_predictor_bmodel_file=calc_predictor_f16.bmodel 
gen_dynamic_fp16bmodel $calc_predictor_model_name $calc_predictor_onnx_file $calc_predictor_input_shapes $calc_predictor_bmodel_file

calc_predictor_model_name=calc_predictor
calc_predictor_onnx_file=onnx/calc_predictor.onnx
calc_predictor_input_shapes=[[1,1000,512]]
calc_predictor_bmodel_file=calc_predictor_f32.bmodel 
gen_dynamic_fp32bmodel $calc_predictor_model_name $calc_predictor_onnx_file $calc_predictor_input_shapes $calc_predictor_bmodel_file

# static
# decoder_model_name=decoder
# decoder_onnx_file=onnx/decoder.onnx
# decoder_input_shapes=[[1,1000,512],[1,600,512],[1,1,1000],[1,600,1]]
# decoder_bmodel_file=decoder_f16.bmodel 
# gen_static_fp16bmodel $decoder_model_name $decoder_onnx_file $decoder_input_shapes $decoder_bmodel_file
# decoder_model_name=decoder
# decoder_onnx_file=onnx/decoder.onnx
# decoder_input_shapes=[[1,1000,512],[1,600,512],[1,1,1000],[1,600,1]]
# decoder_bmodel_file=decoder_f32.bmodel 
# gen_static_fp32bmodel $decoder_model_name $decoder_onnx_file $decoder_input_shapes $decoder_bmodel_file

# dynamic
decoder_model_name=decoder
decoder_onnx_file=onnx/decoder.onnx
decoder_input_shapes=[[1,1000,512],[1,600,512]]
decoder_bmodel_file=decoder_f16.bmodel 
gen_dynamic_fp16bmodel $decoder_model_name $decoder_onnx_file $decoder_input_shapes $decoder_bmodel_file

decoder_model_name=decoder
decoder_onnx_file=onnx/decoder.onnx
decoder_input_shapes=[[1,1000,512],[1,600,512]]
decoder_bmodel_file=decoder_f32.bmodel 
gen_dynamic_fp32bmodel $decoder_model_name $decoder_onnx_file $decoder_input_shapes $decoder_bmodel_file

popd