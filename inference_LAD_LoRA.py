import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.io import read_image
import torchvision.transforms as transforms
from peft import get_peft_model, LoraConfig
from transformers import AutoModel
import numpy as np
import time

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

label_std = [0.000668128 ,0.001756347, 0.002387748, 0.002789993, 0.003103689, 0.003358083, 0.003511276, 0.003699719, 0.003912958, 0.004091768, 0.004288691, 0.00452511,  0.004649166, 0.004755058, 0.004953348, 0.00510443,  0.005143211, 0.00516057 , 0.005276722 ,0.00551354,  0.005661765 ,0.005842991, 0.005988667 ,0.006350857, 0.007181645 ,0.008766369, 0.011425621, 0.016171295, 0.025272556 ,0.034259517]
label_mean = [0.000902081, 0.002551463,0.003894045, 0.004965724, 0.005838002, 0.006643721, 0.007409605, 0.008084464, 0.008751539, 0.009445313, 0.01025541,  0.011159409, 0.012136571, 0.013426239, 0.014955842, 0.0166258, 0.018478586, 0.020649994, 0.023234323, 0.026420219,0.030239716, 0.03512513,0.041234662, 0.049154465 ,0.05964352 , 0.073420364 ,0.091085841 ,0.111922663, 0.13365661 , 0.148688681]
label_mean = torch.tensor(label_mean, dtype=torch.float64)
label_std = torch.tensor(label_std, dtype=torch.float64)

# LoRA 配置
config = LoraConfig(
    inference_mode=True,  # 设为True，表示推理模式
    r=128,  # rank
    lora_alpha=256,  # 缩放因子
    lora_dropout=0.1,  # dropout
    target_modules = ['query', 'value']  # 只在这些模块中应用LoRA
)

def inv_normalize(tensor: torch.Tensor, mean:torch.Tensor, std:torch.Tensor):
    tensor.mul_(std).add_(mean)
    return tensor

# 加载DinoV2的预训练模型
class DinoV2WithDense(nn.Module):
    def __init__(self, 
                 pretrained_model_path="/home/hanb/Desktop/Vit/zhengzhu/dinov2_wheat/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c/",
                 num_classes=30):
        super(DinoV2WithDense, self).__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_model_name)
        
        # 修改后的结构
        self.dense1 = nn.Linear(self.backbone.config.hidden_size, 512)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.dense2 = nn.Linear(512, 256)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.backbone(x).last_hidden_state
        x = features[:, 0]  # 取CLS token
        x = self.dense1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)
        
        return self.classifier(x)

lora_model = DinoV2WithDense(num_classes=30)
lora_model.backbone = get_peft_model(lora_model.backbone, config)

# 冻结 backbone 的参数
for param in lora_model.backbone.parameters():
    param.requires_grad = False  # 冻结 backbone

# 解冻 LoRA 参数
for name, param in lora_model.backbone.named_parameters():
    if "lora" in name:
        param.requires_grad = True  # 解冻 LoRA 参数

# 确保新增 Dense 层的参数可训练
for param in lora_model.dense1.parameters():
    param.requires_grad = True
for param in lora_model.gelu1.parameters():
    param.requires_grad = True  
for param in lora_model.dropout1.parameters():
    param.requires_grad = True  

for param in lora_model.dense2.parameters():
    param.requires_grad = True
for param in lora_model.gelu2.parameters():
    param.requires_grad = True  
for param in lora_model.dropout2.parameters():
    param.requires_grad = True  

for param in lora_model.classifier.parameters():
    param.requires_grad = True

# 数据预处理与测试集
std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

TestTransform = transforms.Compose([
    transforms.Resize([224, 224], interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Normalize(mean, std),
])

class ProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        img_name = os.listdir(self.img_dir)[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = read_image(img_path) / 255.0
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name

# 测试集
test_dataset = ProcessedDataset(
    img_dir=r"/home/hanb/Desktop/Vit/zhengzhu/zong/rice_aia/sanya_gxe/23/20230207/",  # 这里替换成图片文件夹路径
    transform=TestTransform
)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 加载保存的LoRA微调模型
model_path = './LoRA_ckpt/1226_bigwbg_1e3/1226_128_256_s2rbest_model_epoch_100.pth'
checkpoint = torch.load(model_path)

# 去掉 'module.' 前缀此处专门针对训练时候使用了并算！
state_dict = checkpoint['model_state_dict']
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace('module.', '')
    new_state_dict[new_key] = value

lora_model.load_state_dict(new_state_dict)

# 如果模型被包装在 DataParallel 中，去掉它
if isinstance(lora_model, nn.DataParallel):
    lora_model = lora_model.module

# 将模型移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lora_model.to(device)

# 设置模型为评估模式
lora_model.eval()

# 存储预测结果和图像名称
predict = []
image_names = []

# 遍历测试数据集
for i, (inputs, img_names) in enumerate(test_loader, 0):
    # 将数据移到GPU
    inputs = inputs.to(device)

    with torch.no_grad():  # 在推理时不需要计算梯度
        outputs = lora_model(inputs)

    # 反归一化预测值
    predict_batch = inv_normalize(outputs.cpu(), mean=label_mean, std=label_std).cpu().numpy()
    
    # 累积预测值和图像名称
    predict.append(predict_batch)
    image_names.extend(img_names)

# 合并预测值
predict = np.vstack(predict)

# 保存结果到csv文件
results = np.hstack([np.array(image_names).reshape(-1, 1), predict])
df = pd.DataFrame(results)
df.to_csv(r'/home/hanb/Desktop/Vit/zhengzhu/zong/rice_aia/sanya_gxe/23/0207.csv', index=False, header=False)

print("Inference completed. Predictions saved.")
