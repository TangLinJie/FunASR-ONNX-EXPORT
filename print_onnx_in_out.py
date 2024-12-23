from pprint import pprint
import onnxruntime

onnx_path = "models/onnx/decoder.onnx"

onnx_session = onnxruntime.InferenceSession(onnx_path)

print("----------------- 输入部分 -----------------")
input_tensors = onnx_session.get_inputs()  # 该 API 会返回列表
for input_tensor in input_tensors:         # 因为可能有多个输入，所以为列表
    
    input_info = {
        "name" : input_tensor.name,
        "type" : input_tensor.type,
        "shape": input_tensor.shape,
    }
    pprint(input_info)

print("----------------- 输出部分 -----------------")
output_tensors = onnx_session.get_outputs()  # 该 API 会返回列表
for output_tensor in output_tensors:         # 因为可能有多个输出，所以为列表
    
    output_info = {
        "name" : output_tensor.name,
        "type" : output_tensor.type,
        "shape": output_tensor.shape,
    }
    pprint(output_info)