from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import asyncio
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('optimum/all-MiniLM-L6-v2')

# Load the ONNX model
onnx_model_path = "onnx/model.onnx"

# note: for bool type options in python API, set them as False/True
providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_max_workspace_size': 2147483648,
        'trt_fp16_enable': True,
    }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    })
]

sess_opt = ort.SessionOptions()
ort_session = ort.InferenceSession(onnx_model_path, sess_options=sess_opt, providers=providers)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


class InputText(BaseModel):
    text_list: List[str]

async def make_predictions_onnx(input_ids,attention_mask,token_type_ids):
    ort_inputs = {
        'input_ids': input_ids.cpu().numpy(),
        'attention_mask': attention_mask.cpu().numpy(),
        "token_type_ids":token_type_ids.cpu().numpy()
    }
    ort_outputs = ort_session.run(None, ort_inputs)
    return torch.tensor(ort_outputs[0]), attention_mask.cpu().numpy()


async def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


async def process_text(text):
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        model_output=await make_predictions_onnx(encoded_input['input_ids'],encoded_input['attention_mask'],encoded_input['token_type_ids'])
        sentence_embeddings = await mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()

import time 

@app.post("/predict/")
async def predict_texts(input_data: InputText):
    input_texts = input_data.text_list
    print(input_texts)
    
    start_time = time.time()  # Record the start time
    
    tasks = [process_text(text) for text in input_texts]
    print("here")
    results = await asyncio.gather(*tasks)
    
    elapsed_time = (time.time() - start_time) * 1000  # Calculate overall elapsed time in milliseconds
    print(f"Overall time taken: {elapsed_time:.2f} ms")
    return results
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



From nvcr.io/nvidia/tensorrt:22.08-py3

COPY ./app.py ./ 

COPY ./onnx ./onnx

RUN pip install onnxruntime_gpu==1.12.0 uvicorn transformers fastapi 


import requests
import torch
# Replace this URL with the appropriate endpoint if you are running on a different host or port
url = "http://172.17.0.2:8000/predict/"

# Example input data
input_data = {"text_list": ["My name is MK"]}

# Send a POST request to the /predict/ endpoint
response = requests.post(url, json=input_data)

# Print the response
print(response.status_code)
if response.status_code == 200:
    for text in response.json():
        print("here")
        print(torch.tensor(text))
