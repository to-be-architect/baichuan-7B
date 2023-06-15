import datetime
import json

import torch
import uvicorn
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM

CUDA_DEVICE = "cuda:0"

app = FastAPI()


@app.post("/chat")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)

    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    inputs = tokenizer(prompt, return_tensors='pt')

    inputs = inputs.to("cuda:0")

    pred = model.generate(**inputs, max_new_tokens=4096, repetition_penalty=1.1)

    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)

    print(response)

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)

    return answer


if __name__ == '__main__':
    try:

        tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7B",
                                                     trust_remote_code=True,
                                                     # device_map="auto",
                                                     torch_dtype=torch.float16
                                                     ).half().cuda(device=CUDA_DEVICE)

        model.eval()

        uvicorn.run(app, host='0.0.0.0', port=7001, workers=1)

    except Exception as e:
        print(e)
