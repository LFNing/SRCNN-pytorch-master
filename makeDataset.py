import glob
import h5py
import numpy as np
import PIL.Image as pImg

# 将rgb图像转换为灰度图像
def rgb2gray(img):
    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]


def setEvalData(lrImgPath, hrImgPath, h5Path):
    """
    将图片格式的数据集转换为h5格式的数据集
    :param lrImgPath: 低分辨率图片格式数据集的存储路径
    :param hrImgPath: 高分辨率图片格式数据集的存储路径
    :param h5Path: h5格式数据集的存储路径
    :return:
    """

    h5_file = h5py.File(h5Path, 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    # 用于存储低分辨率和高分辨率图像
    for i, p in enumerate(sorted(glob.glob(f'{lrImgPath}/*.*'))):
        lr = pImg.open(p).convert('RGB')
        lr = np.array(lr).astype(np.float32)
        lr = rgb2gray(lr)
        lr_group.create_dataset(str(i), data=lr)

    for i, p in enumerate(sorted(glob.glob(f'{hrImgPath}/*.*'))):
        hr = pImg.open(p).convert('RGB')
        hr = np.array(hr).astype(np.float32)
        hr = rgb2gray(hr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()

if __name__ == '__main__':
    setEvalData(lrImgPath="F:\\CNN_data\\Third\\eval\\lr", hrImgPath="F:\\CNN_data\\Third\\eval\\hr", h5Path=".\\forthEvalData.h5")
