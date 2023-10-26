import gradio as gr
from PIL import Image, ImageDraw
import json
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def inference(img: Image) -> json:
    model_id="wanbiao/resnet18"
    plantClassification = pipeline(Tasks.plant_classification, model_id)
    p,l = plantClassification(img)
    result = {}
    for idx in range(len(l)):
        result[l[idx]] = p[0][idx].item()/100

    print(result)
    return result

title = "万彪的30种植物分类实验基于resnet18"
description = "输入一张图片，输出这张图片所属的种类和概率。"
examples = ['./eggplant.jpg','aloevera.jpg','bilimbi.jpg','paddy.jpg']
outputs = gr.outputs.Label(num_top_classes=5)

demo = gr.Interface(
    fn=inference,
    inputs=gr.inputs.Image(type="filepath"),
    outputs=outputs,
    title=title,
    description=description,
    examples=examples)

demo.launch()