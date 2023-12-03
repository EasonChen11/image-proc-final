import math
from PIL import Image
import optparse
import numpy as np
from numpy import fft as fft
import matplotlib.pyplot as plt
import matplotlib.image as img
carrier_fft=[]
def embed_info(data, x, y, width, layer, flow, count):# 嵌入信息,data为载体图像,x,y为当前坐标,width为当前层的宽度,layer为当前层数,flow为信息流,count为当前信息流的位置
    N = width - 2 * layer - 1
    if N <= 0 or count >= flow.size:
        return data, count
    # 向右嵌入
    for i in range(N):
        if count < flow.size:
            data[y, x+i] = flow[count]
            count += 1
        else:
            return data, count
    
    # 向下嵌入
    for i in range(N):
        if count < flow.size:
            data[y+i, x+N] = flow[count]
            count += 1
        else:
            return data, count
    
    # 向左嵌入
    for i in range(N):
        if count < flow.size:
            data[y+N, x+N-i] = flow[count]
            count += 1
        else:
            return data, count
    
    # 向上嵌入
    for i in range(N):
        if count < flow.size:
            data[y+N-i, x] = flow[count]
            count += 1
        else:
            return data, count
    
    # 递归进入下一层
    return embed_info(data, x+1, y+1, width, layer+1, flow, count)

def extract_info(data, width, flow_size):
    extracted_data = np.zeros(flow_size, dtype=complex)  # 创建一个空数组用于存放提取的数据
    count = 0
    layer = 0
    while count < flow_size:
        N = width - 2 * layer - 1
        if N <= 0:
            break

        x, y = layer, layer

        # 向右提取
        for i in range(N):
            if count < flow_size:
                # if count == 0 or count==1:
                #     print(extracted_data[0])
                extracted_data[count] = data[y, x+i]
                count += 1
            else:
                return extracted_data
        # 向下提取
        for i in range(N):
            if count < flow_size:
                extracted_data[count] = data[y+i, x+N]
                count += 1
            else:
                return extracted_data

        # 向左提取
        for i in range(N):
            if count < flow_size:
                extracted_data[count] = data[y+N, x+N-i]
                count += 1
            else:
                return extracted_data

        # 向上提取
        for i in range(N):
            if count < flow_size:
                extracted_data[count] = data[y+N-i, x]
                count += 1
            else:
                return extracted_data

        layer += 1

    return extracted_data

# 提取过程
def extract_process():
    global carrier_fft
    carrier_img = Image.open("recovered_image.png").convert("L")
    carrier_data = np.array(carrier_img)
    carrier_fft = fft.fftshift(fft.fft2(carrier_data))

    # 假设提取的数据大小和原信息图像的大小一致
    info_img = Image.open("hatwoman.png").convert("L")
    info_data = np.array(info_img)
    info_size = info_data.size

    # 提取信息
    extracted_fft = extract_info(carrier_fft, carrier_data.shape[1], info_size)

    # compare take information array and origin array to see whether the information has been extracted
    print("提取的信息：", extracted_fft)
    print("原信息：",(fft.fft2(info_data)).ravel())

    if np.array_equal(extracted_fft, (fft.fft2(info_data)).ravel()):
        print("提取信息成功")
    else:
        print("提取信息失败")

    # 进行IFFT变换以恢复信息图像
    extracted_data = np.abs(fft.ifft2(extracted_fft.reshape(info_img.size)).round())

    # 显示恢复的信息图像
    plt.imshow(extracted_data, cmap='gray')
    plt.title('Extracted Image')
    plt.show()

    # 保存恢复的信息图像
    Image.fromarray(extracted_data.astype(np.uint8)).save("extracted_image.png")

# 主程序
def main():
    global carrier_fft
    # 载入图像并转换为灰度
    carrier_img = Image.open("camaraman.png").convert("L")
    info_img = Image.open("hatwoman.png").convert("L")

    # 将图像转换为二维数组并进行FFT变换
    carrier_data = np.array(carrier_img)
    info_data = np.array(info_img)
    carrier_fft = fft.fftshift(fft.fft2(carrier_data))
    info_fft = fft.fft2(info_data)
    # 嵌入信息
    carrier_fft, count = embed_info(carrier_fft, 0, 0, carrier_data.shape[1], 0, info_fft.ravel(), 0)
    print("嵌入信息量：", count)
    # compare whether all the data has been embedded
    if count < info_fft.size:
        print("载体图像过小，无法嵌入信息")
    else:
        print("嵌入信息成功")
    # 进行IFFT变换以恢复图像，使用四捨五入的方法
    recovered_data = np.abs(fft.ifft2(fft.ifftshift(carrier_fft)))
    # 显示恢复的图像compare original image and recovered image
    plt.subplot(1, 2, 1)
    plt.imshow(carrier_data, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(recovered_data, cmap='gray')
    plt.title('Recovered Image')
    plt.show()

    # 保存恢复的图像
    Image.fromarray(recovered_data.astype(np.uint8)).save("recovered_image.png")


if __name__ == '__main__':
    main()
    extract_process()