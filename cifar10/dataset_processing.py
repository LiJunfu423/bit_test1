import os
import imageio
import numpy as np

from DataLoader import unpickle, resize_image, image_to_tensor, normalize

train_path = './dataset/cifar-10-batches-py/train'
test_path = './dataset/cifar-10-batches-py/valid'

for i in range(10):
    file_name = train_path + '/' + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

for i in range(10):
    file_name = test_path + '/' + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)
root_dir = "./dataset/cifar-10-batches-py"
print('loading_train_data_')
for j in range(1, 6):
    dataName = root_dir + "/data_batch_" + str(j)  # 读取当前目录下的data_batch1~5文件。
    Xtr = unpickle(dataName)
    print(dataName + " is loading...")

    for i in range(0, 10000):
        img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
        img = img.transpose(1, 2, 0)  # 读取image
        image = resize_image(img)  # 调整图像大小
        tensor_image = image_to_tensor(image)  # 转换为张量
        normalized_image = normalize(tensor_image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 标准化
        picName = root_dir + '/train/' + str(Xtr['labels'][i]) + '/' + str(i + (j - 1) * 10000) + '.jpg'
        imageio.imsave(picName, img)  # 使用的imageio的imsave类
    print(dataName + " loaded.")

# 生成测试集图片(将测试集作为验证集)
print('loading_val_data_')
testXtr = unpickle(root_dir + "/test_batch")
for i in range(0, 10000):
    img = np.reshape(testXtr['data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = root_dir + '/valid/' + str(testXtr['labels'][i]) + '/' + str(i) + '.jpg'
    imageio.imsave(picName, img)