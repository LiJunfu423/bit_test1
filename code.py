import numpy as np
from matplotlib import pyplot as plt


def plot_graph(plot_type):
    # Read data from the uploaded file
    if plot_type == "Training_Loss":
        txt_path = './train_loss.txt'
    elif plot_type == "Validation_Loss":
        txt_path = './valid_loss.txt'
    elif plot_type == "Validation_Accuracy":
        txt_path = './valid_accuracy.txt'
    with open(txt_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")  # Assume data is in the format [value, value, ...]

    y = np.asfarray(data, float)
    x = range(len(y))

    # dn

    plt.figure(figsize=(8, 8))
    if plot_type == "Training_Loss":
        plt.plot(x, y, linewidth=1, linestyle="solid", label='Training Loss')
        plt.title('Training Loss')
    elif plot_type == "Validation_Loss":
        plt.plot(x, y, linewidth=1, linestyle="solid", label='Validation Loss')
        plt.title('Validation Loss')
    elif plot_type == "Validation_Accuracy":
        plt.plot(x, y, linewidth=1, linestyle="solid", label='Validation Accuracy')
        plt.title('Validation Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig("./" + plot_type + ".png")
    plt.close()


def plot_outcome():
    plot_graph("Training_Loss")
    plot_graph("Validation_Loss")
    plot_graph("Validation_Accuracy")

