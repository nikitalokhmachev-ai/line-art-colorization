import gradio as gr
from xdog import to_sketch
from model import Generator, ResNeXtBottleneck
import torch
from data_utils import *
import glob
gen = torch.load('model/model.pth')

def convert_to_lineart(img, sigma, k, gamma, epsilon, phi, area_min):
    phi = 10 * phi
    out = to_sketch(img, sigma=sigma, k=k, gamma=gamma, epsilon=epsilon, phi=phi, area_min=area_min)
    return out

def inference(sk):
    return predict_img(gen, sk, hnt = None)

title = "To Line Art"
description = "Line art colorization showcase. "
article = "Github Repo"

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", value='examples/Genshin-Impact-anime.jpg')
            to_lineart_button = gr.Button("To Lineart")

            gr.Examples(
                examples=glob.glob('examples/*.jpg'),
                inputs=image,
                outputs=image,
                fn=None,
                cache_examples=False,
            )

        with gr.Column():
            sigma = gr.Slider(0.1, 0.5, value=0.3, step=0.1, label='σ')
            k = gr.Slider(1.0, 8.0, value=4.5, step=0.5, label='k')
            gamma = gr.Slider(0.05, 1.0, value=0.95, step=0.05, label='γ')
            epsilon = gr.Slider(-2, 2, value=-1, step=0.5, label='ε')
            phi = gr.Slider(10, 20, label = 'φ', value=15)
            min_area = gr.Slider(1, 5, value=2, step=1, label='Minimal Area')
        
        with gr.Column():
            lineart = gr.Image(type="pil", image_mode='L')
            inpaint_button = gr.Button("Inpaint")

    to_lineart_button.click(convert_to_lineart, inputs=[image, sigma, k, gamma, epsilon, phi, min_area], outputs=lineart)
    inpaint_button.click(inference, inputs=lineart, outputs=lineart)

demo.launch()