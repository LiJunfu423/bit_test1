
import pickle
import numpy as np
import torch
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='iso-8859-1')
        fo.close()
    return dict
def load_cifar10_batch(file):
    data_dict = unpickle(file)
    images = data_dict[b'data']
    labels = data_dict[b'labels']
    # Reshape and transpose the images data
    images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels
def load_label_names(meta_file):
    meta_dict = unpickle(meta_file)
    label_names = meta_dict[b'label_names']
    return [name.decode('utf-8') for name in label_names]
def convert_to_image(images):
    return [Image.fromarray(img) for img in images]
def create_datasets(data_batches, valid_size=0.2):
    train_images = []
    train_labels = []
    for batch in data_batches[:-1]:  # Assume the last batch is for testing
        images, labels = load_cifar10_batch(batch)
        train_images.append(images)
        train_labels.append(labels)
    train_images = np.vstack(train_images)
    train_labels = np.hstack(train_labels)

    test_images, test_labels = load_cifar10_batch(data_batches[-1])

    # Shuffle training data
    indices = np.arange(len(train_images))
    np.random.shuffle(indices)
    train_images = train_images[indices]
    train_labels = train_labels[indices]

    # Split validation data
    split = int(len(train_images) * (1 - valid_size))
    valid_images, valid_labels = train_images[split:], train_labels[split:]
    train_images, train_labels = train_images[:split], train_labels[:split]

    return (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels)
def resize_image(image, size=(224, 224)):
    """调整图像大小到指定尺寸"""
    return image.resize(size, Image.ANTIALIAS)

def image_to_tensor(image):
    """将PIL图像转换为PyTorch张量"""
    numpy_image = np.array(image)
    tensor_image = torch.from_numpy(numpy_image).float()
    tensor_image = tensor_image.permute(2, 0, 1) / 255  # 将通道数从(H, W, C)变为(C, H, W) 并归一化到[0, 1]
    return tensor_image

def normalize(tensor, mean, std):
    """对张量进行标准化"""
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor
