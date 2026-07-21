小土堆 PyTorch TensorBoard 完整课程笔记
小土堆 PyTorch TensorBoard 完整课程笔记
一、TensorBoard 工具作用
TensorBoard 是深度学习可视化工具，解决训练过程 “黑盒” 问题，可直观查看：损失 / 准确率曲线、数据集图片样本、网络结构、参数分布。
课程分为两节：
1. 第一节：安装、SummaryWriter、add_scalar 绘制标量曲线
2. 第二节：add_image 图片可视化、图片通道格式避坑

---
第一节 SummaryWriter 与 add_scalar
1. 安装依赖
虚拟环境终端执行
pip install tensorboard
2. 核心类 SummaryWriter
2.1 导入模块
from torch.utils.tensorboard import SummaryWriter
2.2 实例化对象
# 传入字符串=日志文件夹名，代码运行自动创建
writer = SummaryWriter("logs")
- 文件夹作用：存放事件日志文件 events.out.tfevents.xxxx
- 规范：不同实验用不同文件夹名，防止曲线重叠混淆
2.3 必写操作：writer.close()
程序末尾必须关闭，缓存数据完整写入本地，否则网页无数据、日志缺失。
3. add_scalar 绘制曲线（记录标量：loss/acc/ 自定义函数）
3. 函数参数
writer.add_scalar(tag, scalar_value, global_step)
参数
含义
tag
曲线名称（字符串），同名会合并曲线
scalar_value
Y 轴数值（损失、精度、i、2*i 等）
global_step
X 轴步数（迭代次数 /epoch）
完整示例代码
from torch.utils.tensorboard import SummaryWriter

# 1. 初始化日志写入器
writer = SummaryWriter("logs")

# 2. 绘制 y=x 曲线，x范围 0~99
for i in range(100):
    writer.add_scalar(tag="y=x", scalar_value=i, global_step=i)

# 3. 关闭写入缓存
writer.close()
4. 启动网页可视化
基础启动命令
tensorboard --logdir=logs
端口占用时指定端口
tensorboard --logdir=logs --port=6007
- logdir：只填日志文件夹名称，不能填具体事件文件
- 默认访问地址：http://localhost:6006
- 更新数据后刷新浏览器即可加载新曲线
5. 第一节高频避坑
1. 多次运行代码曲线重叠：运行前手动删除 logs 文件夹
2. 忘记 writer.close()：日志不完整，网页空白无曲线
3. 多条曲线使用相同 tag：数据叠加在同一条曲线上，无法区分
4. 虚拟环境不匹配：安装、运行 tensorboard 必须在同一个 conda 环境

---
第二节 add_image 图片可视化
1. add_image 作用
查看训练集 / 测试集图片样本，校验数据预处理、图像通道是否正确。
函数参数
writer.add_image(tag, img_tensor, global_step, dataformats="CHW")
参数
说明
tag
图片分组名称（train/test 分开）
img_tensor
图像数据，支持 numpy 数组 /torch.Tensor
global_step
步数，拖动滑块切换不同 step 图片
dataformats
通道顺序（最容易报错）
- CHW：Tensor 格式（通道、高、宽）ToTensor 转换后默认
- HWC：numpy 数组格式（高、宽、通道）PIL/OpenCV 读取默认
2. 两种图片读取完整代码
案例 1：PIL 读取图片 → numpy 数组（HWC，必须指定格式）
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
# 图片路径
img_path = "dataset1/train/ants/0013035.jpg"
img_pil = Image.open(img_path)
img_array = np.array(img_pil)  # shape (H, W, 3)

# numpy数组必须加 dataformats="HWC"
writer.add_image(tag="train", img_array, global_step=1, dataformats="HWC")

writer.close()
案例 2：ToTensor 转为张量（CHW，默认无需修改格式）
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from torchvision import transforms

writer = SummaryWriter("logs")
img_path = "dataset1/train/ants/0013035.jpg"
img_pil = Image.open(img_path)

# 转为 Tensor (3, H, W)
trans = transforms.ToTensor()
img_tensor = trans(img_pil)

# 默认 CHW，不用传dataformats
writer.add_image("train_tensor", img_tensor, global_step=1)
writer.close()
3. 多步图片滑动查看（循环写入不同 step）
同一 tag 下多次调用 add_image，网页出现滑块切换图片
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
img_path = "dataset1/train/ants/0013035.jpg"
img_pil = Image.open(img_path)
img_array = np.array(img_pil)

# step 1、2、3 存入同一张图，网页可滑动切换
for step in [1, 2, 3]:
    writer.add_image("train", img_array, global_step=step, dataformats="HWC")

writer.close()
4. 第二节核心避坑
1. numpy 图片不写 dataformats="HWC"：图片颜色颠倒、画面发黑
2. 像素数值超出范围：numpy 图片 0\255，Tensor 图片 0\1，失真花屏
3. 灰度单通道图：dataformats="HW"
4. 批量多张图片网格展示：使用 add_images() + torchvision.utils.make_grid()
5. 网页图片显示规则
- 仅调用 1 次 add_image：页面只显示 1 张图，无滑动条
- 多次调用、不同 global_step：页面出现滑块，拖动切换不同迭代图片
- 残留旧图片分组：删除本地 logs 文件夹清空历史数据

---
完整综合可运行模板
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from torchvision import transforms

# 初始化日志
writer = SummaryWriter("logs")

# 1. 绘制标量曲线 y=2x
for i in range(100):
    writer.add_scalar("y=2x", scalar_value=2*i, global_step=i)

# 2. 写入图片
img = Image.open("dataset1/train/ants/0013035.jpg")
img_arr = np.array(img)
writer.add_image("train_ant", img_arr, global_step=0, dataformats="HWC")

# 关闭缓存
writer.close()

---
补充实操踩坑记录（protobuf 兼容）
1. 高版本 protobuf 报错 cannot import name 'runtime_version'
  - 方案 1：用 Python 代码启动 TensorBoard，不使用终端命令
  - 方案 2：降级 protobuf 至 6.32.0，重启终端执行启动命令
2. 终端命令启动报错，但 Python 脚本写入日志正常：仅终端 exe 启动器存在库缓存冲突，优先代码内启动面板
