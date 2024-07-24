import gradio as gr
import numpy as np
from matplotlib import pyplot as plt
from predict import classify_image


def plot_graph(plot_type):
    # Read data from the uploaded file
    if plot_type == "Training Loss":
        txt_path = './train_loss.txt'
    elif plot_type == "Validation Loss":
        txt_path = './valid_loss.txt'
    elif plot_type == "Validation Accuracy":
        txt_path = './valid_accuracy.txt'
    with open(txt_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")  # Assume data is in the format [value, value, ...]

    y = np.asfarray(data, float)
    x = range(len(y))

    plt.figure(figsize=(8, 8))
    if plot_type == "Training Loss":
        plt.plot(x, y, linewidth=1, linestyle="solid", label='Training Loss')
        plt.title('Training Loss')
    elif plot_type == "Validation Loss":
        plt.plot(x, y, linewidth=1, linestyle="solid", label='Validation Loss')
        plt.title('Validation Loss')
    elif plot_type == "Validation Accuracy":
        plt.plot(x, y, linewidth=1, linestyle="solid", label='Validation Accuracy')
        plt.title('Validation Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    return plt.gcf()


demo = gr.Interface(
    fn=plot_graph,
    inputs=gr.Dropdown(choices=["Training Loss", "Validation Loss", "Validation Accuracy"]),
    outputs="plot",
    description="选择Training Loss, Validation Loss,  Validation Accuracy图."
)
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(label="上传图片"),
    outputs=gr.Text(label="预测结果"),
    title="图像分类预测",
    description="请上传图片"
)
app = gr.TabbedInterface([iface, demo], ["图像预测", "数据可视化"])
app.launch()