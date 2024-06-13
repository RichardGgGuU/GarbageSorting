import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def load_random_images(img_dir, num_images=5):
    img_files = [file for file in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, file))]
    img_files = np.random.choice(img_files, num_images, replace=False)
    return [os.path.join(img_dir, img_file) for img_file in img_files]

def show_processed_images(img_dir, num_images=5):
    img_paths = load_random_images(img_dir, num_images)
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(150, 150))
        plt.imshow(img)
        plt.title('Processed Image')
        plt.show()

def main():
    test_dir = 'RubbishData_full/test'  # 测试图像目录路径
    num_images = 5  # 要展示的图像数量

    # 展示处理后的图像
    show_processed_images(test_dir, num_images)

if __name__ == '__main__':
    main()
