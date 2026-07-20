Pytorch 小土堆课程 P6+P7 完整学习笔记（Dataset 数据加载实战）
markdown
# 小土堆PyTorch P6+P7 全套学习笔记
## 一、P6（数据加载初识）核心理论回顾
### 1. Dataset 与 DataLoader 核心分工
1. `torch.utils.data.Dataset`：**单样本读取层**，仅根据索引返回单条数据「图片+标签」
2. `torch.utils.data.DataLoader`：**批量打包层**，实现分批加载、数据打乱、多线程读取，为神经网络提供batch批量数据

### 2. 自定义Dataset类必须实现的3个魔法函数
| 方法 | 核心作用 |
|------|------|
| `__init__(self, root_dir)` | 初始化：保存数据集根目录，预读取全部图片文件名 |
| `__getitem__(self, idx)` | 按下标索引获取单张样本，拼接路径、读取图片、匹配标签 |
| `__len__(self)` | 返回数据集总样本数量，支持 `len(数据集实例)` 调用 |

### 3. 路径处理工具函数（解决路径报错关键）
- `os.listdir(folder_path)`：读取指定文件夹内所有文件名称，生成文件名列表，用于遍历数据集图片
- `os.path.join(path1, path2)`：跨系统安全拼接文件路径，自动适配Windows/Linux分隔符，规避转义字符、`\u202a`隐藏控制字符报错

### 4. 数据集拓展特性
数据集对象支持直接使用 `+` 运算符合并，快速拼接多分类/多组数据集：
```python
dataset1 = AntBeeDataset("hymenoptera_data/train/ants")
dataset2 = AntBeeDataset("hymenoptera_data/train/bees")
all_dataset = dataset1 + dataset2
二、P7 实战：手写自定义 Dataset 完整代码 + 必掌握知识点
（一）本节课整体授课内容
基于蚂蚁蜜蜂分类数据集，从零完整手写自定义 Dataset 类
逐行拆解 3 个魔法函数实现逻辑：路径存储、图片读取、标签自动匹配
实例化数据集，测试索引取值、统计样本总数量
搭配 DataLoader 实现批量数据加载，为下一节 TensorBoard 可视化做数据铺垫
（二）核心可运行完整代码
python
运行
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# 1. 自定义图像分类数据集类
class AntBeeDataset(Dataset):
    def __init__(self, root_dir):
        # root_dir：数据集train/val文件夹根路径，如 "hymenoptera_data/train"
        self.root_dir = root_dir
        # 分类文件夹列表，对应两类标签
        self.class_dirs = ["ants", "bees"]
        # 列表存储元组：(图片完整路径, 对应数字标签)
        self.img_path_list = []

        # 遍历分类文件夹，自动分配标签
        for label, cls_name in enumerate(self.class_dirs):
            cls_full_path = os.path.join(self.root_dir, cls_name)
            img_names = os.listdir(cls_full_path)
            for img_name in img_names:
                img_full_path = os.path.join(cls_full_path, img_name)
                self.img_path_list.append((img_full_path, label))

    def __getitem__(self, idx):
        # 根据索引取出单张图片路径与标签
        img_path, label = self.img_path_list[idx]
        # PIL读取图片，返回图像对象
        img = Image.open(img_path)
        # 统一返回格式：(图像, 标签)
        return img, label

    def __len__(self):
        # 返回数据集总样本数量
        return len(self.img_path_list)

# 2. 实例化训练集
train_dataset = AntBeeDataset(root_dir="hymenoptera_data/train")

# 3. 基础功能测试
# 取第0号样本
img, target = train_dataset[0]
print(f"图片尺寸: {img.size}, 对应标签: {target}")
# 统计训练集总图片数量
print(f"训练集样本总数: {len(train_dataset)}")

# 4. DataLoader 批量加载（训练核心）
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,       # 单批次读取图片数量
    shuffle=True,       # 每个epoch打乱数据顺序（训练集开启，验证集关闭）
    num_workers=0       # Windows系统必须设为0，Linux可开启多线程加速
)

# 遍历批量数据测试
for imgs, labels in train_loader:
    print(f"一批图像张量shape: {imgs.shape}, 批次标签列表: {labels}")
    break
（三）分模块核心知识点拆解
1. __init__ 初始化模块要点
enumerate 自动给分类文件夹分配数字标签：ants=0，bees=1，是图像分类任务标准写法
预存储全部图片路径与标签，避免每次索引重复遍历文件夹，提升读取效率
全程使用 os.path.join 拼接路径，兼容全系统，杜绝路径转义、隐藏字符报错
2. __getitem__ 样本读取模块要点
传入参数 idx 为自动接收的下标，执行 dataset[0] 时会自动调用该函数
Image.open() 读取图片返回 PIL 图像对象，可搭配 torchvision.transforms 做数据增强
强制返回 (图像, 标签) 二元组，是 PyTorch 数据集统一规范，兼容所有模型、可视化工具
3. __len__ 长度模块要点
必须 return 样本总数，否则调用 len(train_dataset) 直接抛出异常
DataLoader、训练循环会依赖该方法计算完整 epoch 迭代次数
4. DataLoader 核心参数对照表
表格
参数名	功能说明	实操注意事项
dataset	传入自定义 Dataset 实例	禁止直接传入文件夹路径，类型不匹配会报错
batch_size	单批次加载图片数量	根据显卡显存调整，常用 4/8/16/32
shuffle	是否打乱样本顺序	训练集 = True，验证 / 测试集 = False
num_workers	多线程加载进程数	Windows 强制写 0；Linux 可设 2/4/8 加速读取
（四）高频实操踩坑点
漏写 __getitem__ / __len__ 函数：索引取值、len() 调用直接报错
手动硬编码路径（E:\dataset\img.jpg），未使用os.path.join：复制路径带入隐藏控制字符，触发OSError路径非法
Windows 环境设置num_workers>0：程序卡死、多进程创建失败
混淆 Dataset 与 DataLoader 作用：直接将文件夹传入 DataLoader，类型不匹配
标签与图片路径存储错位：分类标签匹配错误，模型训练无法收敛
（五）课程前后章节衔接
本节输出的 PIL 图像、批量 Tensor 数据，直接对接下一节 P8（BV1hE411t7RN p7）TensorBoard 可视化：
使用 writer.add_image() 将数据集图片写入可视化面板，实现训练数据在线查看。
三、本节学习达标要求
脱离笔记独立默写自定义 Dataset 三函数基础框架
可自主修改数据集根目录、batch_size、shuffle 等核心参数适配自己的数据集
独立排查三类高频报错：文件路径找不到、权限拒绝、Windows 多进程崩溃
完整理解数据流转链路：本地图片文件 → Dataset 单样本读取 → DataLoader 批量 Tensor → 送入网络 / TensorBoard 可视化