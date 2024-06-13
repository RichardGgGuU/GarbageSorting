import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_random_images(img_dir, num_images=5):
    img_files = [file for file in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, file))]
    img_files = np.random.choice(img_files, num_images, replace=False)
    return [os.path.join(img_dir, img_file) for img_file in img_files]

def predict_images(model, img_paths):
    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
    for img_path in img_paths:
        # 加载图像
        img = image.load_img(img_path, target_size=(150, 150))
        # 转换为数组并进行归一化
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        # 预测
        prediction = model.predict(img_array)
        predicted_label = labels[np.argmax(prediction)]
        # 提取实际分类
        actual_label = os.path.basename(os.path.dirname(img_path))
        # 显示图像和预测结果
        plt.imshow(img)
        plt.title(f'Actual: {actual_label}, Predicted: {predicted_label}')
        plt.show()

def main():
    model_path = 'model1.h5'  # 模型路径
    path = ['RubbishData_full/data_full/cardboard', 'RubbishData_full/data_full/glass',
            'RubbishData_full/data_full/metal', 'RubbishData_full/data_full/paper',
            'RubbishData_full/data_full/plastic',
            'RubbishData_full/data_full/trash']
    for i in range(5):
        test_dir = path[i]
        print(test_dir)  # 测试图像目录路径
        num_images = 5  # 要预测的图像数量
        # 加载模型
        model = load_model(model_path)
        # 随机选择图像并进行预测
        img_paths = load_random_images(test_dir, num_images)
        predict_images(model, img_paths)

if __name__ == '__main__':
    main()
