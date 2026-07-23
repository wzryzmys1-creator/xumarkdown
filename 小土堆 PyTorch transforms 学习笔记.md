# 小土堆 PyTorch transforms 学习笔记

## 一、transforms 整体作用

`torchvision.transforms` 是 PyTorch 官方图像预处理工具包，主要用于深度学习数据集预处理，核心作用有三点：

1. 统一图片尺寸、格式、像素范围，适配神经网络输入规范

2. 实现数据增强（随机裁剪、翻转等），有效缓解模型过拟合

3. 完成数据格式转换：PIL图像/Numpy数组 → 网络可识别的Tensor格式

模块导入代码：

```python
from torchvision import transforms
```

## 二、核心工具：transforms\.Compose\(\)

用于将多个预处理操作打包成一个流水线，**严格按照代码从上到下的顺序依次执行**，是项目中最常用的组合方式。

```python
trans = transforms.Compose([
    操作1,
    操作2,
    操作3
])
```

> 注意：不同预处理操作支持的图像格式不同，流水线顺序绝对不能乱序，否则会直接报错！
> 
> 

## 三、常用 transform 函数详解

### 1\. transforms\.ToTensor\(\)【全网最核心、必用】

深度学习必备转换函数，是图片送入网络的前置操作。

```python
trans = transforms.ToTensor()
img_tensor = trans(pil_img)
```

三大核心功能：

1. 数据类型转换：PIL Image / numpy\.ndarray → torch\.Tensor

2. 通道维度转换：HWC（PIL/数组格式） → CHW（Tensor标准格式）

3. 像素值归一化：0\~255（原图像素范围） → 0\~1（网络适配范围）

⚠️ 避坑：不可对已经是 Tensor 格式的数据重复执行 ToTensor\(\)，会直接报错。

### 2\. transforms\.Resize\(\) 修改图像尺寸

统一数据集图片尺寸，适配网络输入大小，输入输出均为 PIL 图像。

```python
# 写法1：等比例缩放，短边/长边统一缩放到256
trans = transforms.Resize(256)

# 写法2：固定尺寸 (高, 宽)，精准自定义
trans = transforms.Resize((256, 256))
```

### 3\. transforms\.Normalize\(\) 标准化

对 Tensor 图像做均值、方差标准化，提升模型训练收敛速度。

```python
# 依次对应 R、G、B 三个通道的均值、标准差
trans = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
```

计算公式：$output=\frac{input-mean}{std}$

常用效果：当 mean=0\.5、std=0\.5 时，像素区间由 `[0,1]` 映射为 `[-1,1]`

⚠️ 硬性规则：**仅支持 Tensor 格式，必须写在 ToTensor\(\) 之后！**

### 4\. transforms\.CenterCrop\(\) 中心裁剪

从图片正中心裁剪指定尺寸的区域，多用于测试集预处理，输入输出均为 PIL 图像。

```python
# 中心裁剪 256×256 正方形区域
trans = transforms.CenterCrop(256)

# 自定义裁剪尺寸 (高, 宽)
trans = transforms.CenterCrop((256, 512))
```

### 5\. 训练集专属数据增强操作

随机操作仅用于训练集，扩充数据多样性、防止过拟合，**测试集/验证集禁止使用**。

#### RandomCrop 随机裁剪

```python
# 随机裁剪256尺寸，边缘填充4像素
trans = transforms.RandomCrop(256, padding=4)
```

#### RandomHorizontalFlip 随机水平翻转

```python
# p=0.5 代表50%概率随机翻转
trans = transforms.RandomHorizontalFlip(p=0.5)
```

### 6\. transforms\.Pad\(\) 边缘填充

对图片边缘进行像素填充，适配裁剪、尺寸统一需求，输入输出为 PIL 图像。

```python
# 上下左右统一填充2个像素
trans = transforms.Pad(padding=2)
```

## 四、标准预处理执行顺序（固定、不可颠倒）

**PIL原图 → Resize尺寸调整 → 随机数据增强 → ToTensor格式转换 → Normalize标准化**

完整可运行示例代码：

```python
from PIL import Image
from torchvision import transforms

# 搭建预处理流水线
trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# 处理图片
img = Image.open("test.jpg")
img_tensor = trans(img)
print(img_tensor.shape)
```

## 五、高频报错踩坑汇总

1. **顺序错误**：Normalize 写在 ToTensor 之前，Normalize仅支持Tensor，直接报错

2. **格式混淆**

    - PIL图像可用：Resize、CenterCrop、RandomHorizontalFlip、Pad

    - Tensor图像可用：Normalize

    - ToTensor：专属转换工具，PIL/numpy → Tensor

3. **通道参数不匹配**

    - 彩色3通道图：mean、std 必须写3个数值

    - 灰度单通道图：mean、std 只能写1个数值

4. **测试集误用随机增强**：RandomFlip、RandomCrop 仅训练集使用，测试集需固定预处理，避免预测结果不稳定

## 六、测试集标准预处理流水线（无随机操作）

```python
test_trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
```

## 七、结合TensorBoard综合实操代码

预处理图片 \+ TensorBoard可视化，校验预处理效果

```python
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

writer = SummaryWriter("logs")
img = Image.open("dataset1/train/ants/0013035.jpg")

# 图片预处理
trans = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])
img_tensor = trans(img)

# 可视化预处理后的图片
writer.add_image("transform_img", img_tensor, global_step=1)
writer.close()
```

> （注：部分内容可能由 AI 生成）
