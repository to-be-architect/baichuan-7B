import gradio as gr
import mdtex2html

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CUDA_DEVICE = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7B",
                                             trust_remote_code=True,
                                             # device_map="auto",
                                             torch_dtype=torch.float16
                                             ).half().cuda(device=CUDA_DEVICE)

model = model.eval()

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    print(f'输入：{input}')

    chatbot.append((parse_text(input), ""))

    inputs = tokenizer(input, return_tensors='pt')

    # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!
    inputs = inputs.to(CUDA_DEVICE)

    pred = model.generate(**inputs, max_new_tokens=4096, repetition_penalty=1.1)

    response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)

    print(f'输出：{response}')

    chatbot[-1] = (parse_text(input), parse_text(response))

    # 释放GPU内存：在每次模型计算后，您可以使用torch.cuda.empty_cache()方法释放GPU上的内存，以便为后续计算腾出空间。
    torch.cuda.empty_cache()

    yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">BaiChat</h1>""")

        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=12):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                        container=False)

                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("Submit", variant="primary")

            with gr.Column(scale=1):
                emptyBtn = gr.Button("Clear History")
                max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

        history = gr.State([])

        submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                        show_progress=True)

        submitBtn.click(reset_user_input, [], [user_input])

        emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

    demo.queue().launch(share=False, inbrowser=True)
