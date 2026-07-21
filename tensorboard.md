小土堆 PyTorch TensorBoard 两节课程完整知识点总结（Markdown 版）
一、课程整体作用
TensorBoard 是训练可视化工具，解决深度学习 “黑盒训练” 问题，直观查看损失、图片样本、模型结构变化，分为两节：
第一节：基础安装、SummaryWriter、add_scalar 标量曲线绘制
第二节：add_image 图片可视化、图片格式转换、完整避坑实操
第一节 TensorBoard 基础与 add_scalar
1. 环境安装
bash
运行
# 虚拟环境执行
pip install tensorboard
2. 核心类：SummaryWriter
2.1 导包
python
运行
from torch.utils.tensorboard import SummaryWriter
2.2 实例化
python
运行
# 参数：日志文件夹名称，自动创建
writer = SummaryWriter("logs")
文件夹作用：存放事件日志文件（events.out.tfevents.xxx）
命名规范：不同实验分开文件夹，避免曲线重叠
2.3 必须操作：writer.close ()
程序结束一定要关闭，缓存数据完整写入本地日志，否则数据丢失。
3. 核心方法 add_scalar（绘制曲线，记录标量）
3.1 函数参数
python
运行
writer.add_scalar(tag, scalar_value, global_step)
tag：字符串，曲线标题 / 分类（如 train_loss、y=x），不同曲线不能同名
scalar_value：Y 轴数值（损失、准确率、自定义函数值）
global_step：X 轴步数（epoch、迭代次数）
3.2 基础示例代码
python
运行
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")

# 绘制 y=x 曲线，x范围0~99
for i in range(100):
    writer.add_scalar(tag="y=x", scalar_value=i, global_step=i)

writer.close()
4. 启动 TensorBoard 网页查看日志
4.1 终端命令
bash
运行
tensorboard --logdir=logs
# 指定端口（解决端口占用）
tensorboard --logdir=logs --port=6007
logdir：日志文件夹名，不要写具体事件文件
访问地址：默认 http://localhost:6006/
4.2 网页刷新规则
修改代码重跑后，刷新浏览器即可加载新曲线。
5. 第一节高频避坑点
重复运行代码曲线重叠：新实验前手动删除 logs 文件夹
writer.close() 忘记写：日志不完整、网页无数据
tag 相同：多条数据会画在同一条曲线上，造成混乱
虚拟环境不匹配：安装 / 运行 tensorboard 必须在同一个 env
第二节 add_image 图片可视化 & 图片格式规范
1. add_image 核心作用
可视化训练样本、特征图，观察输入图片是否预处理正确。
函数参数
python
运行
writer.add_image(tag, img_tensor, global_step, dataformats="CHW")
tag：图片分组标题
img_tensor：图片数据，支持 PIL转numpy数组 / torch.Tensor
global_step：步数，可滑动切换不同 step 的图片
dataformats：图片通道顺序（最容易报错）
默认 CHW：Tensor 格式（通道、高、宽）
HWC：numpy 数组格式（高、宽、通道，OpenCV/PIL 转数组默认）
2. 两种图片读取实操案例
案例 1：PIL.Image 读取 → numpy 数组（HWC）
python
运行
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
img_path = "test.jpg"
img = Image.open(img_path)
img_np = np.array(img) # shape (H, W, 3)

# 必须指定 dataformats="HWC"
writer.add_image("train_sample", img_np, global_step=1, dataformats="HWC")
writer.close()
案例 2：ToTensor () 转为 Tensor（CHW，默认格式不用改）
python
运行
from torchvision import transforms
trans = transforms.ToTensor()
img_tensor = trans(img) # shape (3, H, W)
writer.add_image("train_sample", img_tensor, global_step=1)
3. 多步图片对比（滑动查看不同 epoch 图片）
循环不同 step 写入，网页滑块切换图片：
python
运行
for step in range(10, 50, 10):
    img_tensor = trans(Image.open(f"img_{step}.jpg"))
    writer.add_image("train_img", img_tensor, global_step=step)
4. 第二节核心避坑（90% 报错来源）
图片 shape 与 dataformats 不匹配：
numpy 数组 (HWC) 不写dataformats="HWC" → 图片颜色错乱、黑屏
图片像素值范围：必须 0~255（numpy）或 0~1（Tensor），超出会失真
单通道灰度图：dataformats 设为HW
批量多张图用 add_images()，配合 torchvision.utils.make_grid() 拼接网格图
两节课程通用完整模板（可直接复制运行）
python
运行
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from torchvision import transforms

# 1. 初始化
writer = SummaryWriter("logs")

# 2. 绘制标量曲线
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

# 3. 写入图片
trans = transforms.ToTensor()
img = Image.open("test.jpg")
img_tensor = trans(img)
writer.add_image("sample", img_tensor, global_step=0)

# 4. 关闭写入
writer.close()
两节课程核心对比速查表
表格
方法	用途	核心参数	训练场景
add_scalar	标量曲线	tag、y 值、step	loss、准确率、学习率
add_image	图片可视化	tag、图片、step、通道格式	样本检查、特征图查看
补充你遇到的 protobuf 兼容问题（课程外实操补充）
高版本 protobuf 会报runtime_version导入错误；
两种绕过方案：
方案 1：Python 代码内启动 tensorboard，不使用终端命令
方案 2：降级 protobuf 至 6.32.0，重启终端后执行tensorboard --logdir=logs