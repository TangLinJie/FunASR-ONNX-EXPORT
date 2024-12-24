"""
from funasr import AutoModel
import time
import multiprocessing 

# setting
dev_id = 0                              ## bm1684x/bm1688 device id
input_path = "LibriSpeech/train-clean-5/7367/86737/7367-86737-0118.flac" # "./20240711090630019.wav" # "./chuanda.wav" #        ## input audio path

def process():
    # offline asr demo
    model = AutoModel(model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
                            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                            # punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                            # spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                            )
    # inference
    start_time = time.time()
    res = model.generate(input=input_path,
                        batch_size_s=300,
                        fs=16000
                        )
    end_time = time.time()
    print(res)
    print("generate time:", end_time-start_time)

process()
"""
import os
out_dirs = 'models/onnx'

if not os.path.exists(out_dirs):
    os.makedirs(out_dirs)


from funasr import AutoModel
import time

chunk_size = 200 # ms
model = AutoModel(model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")

# offline asr demo
asr_model = AutoModel(model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
                        # vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                        # punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                        # spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                        )

import soundfile
import numpy as np

wav_file = f"LibriSpeech/train-clean-5/7367/86737/7367-86737-0118.flac"
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = int(chunk_size * sample_rate / 1000)

"""
speech, sample_rate = soundfile.read(wav_file)
asr_res = asr_model.generate(input=speech,
                                batch_size_s=300,
                                fs=16000
                                )
print(asr_res)
import sys
sys.exit(0)

acc_num = 0
total_num = 0
for audio_root_dir in os.listdir("LibriSpeech/train-clean-5"):
    for audio_root_sub_dir in os.listdir(f"LibriSpeech/train-clean-5/{audio_root_dir}"):
        labels_path = f"LibriSpeech/train-clean-5/{audio_root_dir}/{audio_root_sub_dir}/{audio_root_dir}-{audio_root_sub_dir}.trans.txt"
        with open(labels_path) as f:
            labels = [(line.split(' ')[0], ' '.join(line.strip().split(' ')[1:])) for line in f.readlines()]
        labels = dict(labels)
        for audio_name in os.listdir(f"LibriSpeech/train-clean-5/{audio_root_dir}/{audio_root_sub_dir}"):
            if "flac" in audio_name:
                speech, sample_rate = soundfile.read(f"LibriSpeech/train-clean-5/{audio_root_dir}/{audio_root_sub_dir}/{audio_name}")
                asr_res = asr_model.generate(input=speech,
                                                batch_size_s=300,
                                                fs=16000
                                                )
                if asr_res[0]["text"].upper() == labels[audio_name.replace(".flac", "")]:
                    print("acc")
                    acc_num += 1
                else:
                    print(f"{asr_res[0]['text'].upper()}\n{labels[audio_name.replace('.flac', '')]}")
                total_num += 1
print("acc%: ", acc_num / total_num)
import sys
sys.exit(0)
"""
cache = {}
total_chunk_num = int(len((speech)-1)/chunk_stride+1)
speech_start = False
speech_list = []
all_asr_res = ""
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size,
                        fs=16000)
    if len(res[0]["value"]):
        print(res)


    for ele in res[0]["value"]:
        if ele[0] != -1 and ele[1] == -1:
            speech_start = True
            speech_list.append(speech[int(ele[0]/1000*sample_rate):int(((i+1) * chunk_size) / 1000 * sample_rate)])
        elif ele[0] != -1 and ele[1] != -1:
            # inference
            start_time = time.time()
            input_bytes = speech_chunk
            asr_res = asr_model.generate(input=input_bytes,
                                batch_size_s=300,
                                fs=16000
                                )
            end_time = time.time()
            all_asr_res += ' ' + asr_res[0]["text"]
            print(asr_res)
            print("generate time:", end_time-start_time)
        elif ele[0] == -1:
            speech_start = False
            speech_list.append(speech_chunk[:int((ele[1] - i * chunk_size)/1000*sample_rate)])
            input_bytes = np.concatenate(speech_list)
            speech_list = []
            # inference
            start_time = time.time()
            asr_res = asr_model.generate(input=input_bytes,
                                batch_size_s=300,
                                fs=16000
                                )
            end_time = time.time()
            all_asr_res += ' ' + asr_res[0]["text"]
            print(asr_res)
            print("generate time:", end_time-start_time)
    if len(res[0]["value"]) == 0 and speech_start:
        speech_list.append(speech_chunk)
print(all_asr_res)

"""
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020', model_revision="v2.0.4")

rec_result = inference_pipeline(input='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_en.wav')
print(rec_result)

"""
"""
from funasr import AutoModel
import time
import multiprocessing 

# setting
dev_id = 0                              ## bm1684x/bm1688 device id
input_path = "LibriSpeech/train-clean-5/7367/86737/7367-86737-0118.flac" # "./20240711090630019.wav" # "./chuanda.wav" #        ## input audio path

def process():
    # offline asr demo
    model = AutoModel(model="iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",    ## 语音识别模型
                    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",                             ## 语音端点检测模型
                    punc_model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",                ## 标点恢复模型
                    dev_id=dev_id,
                    )
    # inference
    start_time = time.time()
    res = model.generate(input=input_path,
                        batch_size_s=300,
                        )
    end_time = time.time()
    print(res)
    print("generate time:", end_time-start_time)

process()
"""
"""
from datasets import load_dataset

ds = load_dataset("mozilla-foundation/common_voice_17_0", "en", split="test")
"""
