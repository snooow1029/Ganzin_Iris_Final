import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torchvision.models import resnet50, googlenet

class IrisDataset(Dataset):
    def __init__(self, mode='train', transform=None):
        """
        虹膜验证数据集加载器
        
        参数:
            mode (str): 'train' 或 'test'
            transform: 图像预处理操作
        """
        self.mode = mode
        self.transform = transform
        self.images = []           # 存储图片路径
        self.subject_ids = []      # 存储人物ID
        self.subject_to_images = {}  # 映射subject_id到图像路径列表
        
        # 基础路径
        base_path = r"D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\dataset"
        
        # 加载CASIA-Iris-Lamp数据集
        lamp_path = os.path.join(base_path, "CASIA-Iris-Lamp")
        self._load_lamp_dataset(lamp_path)
        
        # 加载CASIA-Iris-Thousand数据集
        thousand_path = os.path.join(base_path, "CASIA-Iris-Thousand")
        self._load_thousand_dataset(thousand_path)
        
        # 加载Ganzin-J7EF-Gaze数据集
        gaze_path = os.path.join(base_path, "Ganzin-J7EF-Gaze")
        self._load_gaze_dataset(gaze_path)
        
        print(f"数据集加载完成 - {mode} 模式")
        print(f"样本数量: {len(self.images)}")
        print(f"人数: {len(self.subject_to_images)}")
    
    def _load_lamp_dataset(self, lamp_path):
        """加载CASIA-Iris-Lamp数据集"""
        if not os.path.exists(lamp_path):
            print(f"路径不存在: {lamp_path}")
            return
            
        for subject_id in os.listdir(lamp_path):
            subject_path = os.path.join(lamp_path, subject_id)
            
            # 检查是否为文件夹
            if not os.path.isdir(subject_path):
                continue
            
            # 检查是训练集还是测试集
            subject_id_num = int(subject_id)
            is_test = (1 <= subject_id_num <= 80)
            
            # 如果模式与数据集不匹配则跳过
            if (self.mode == 'train' and is_test) or (self.mode == 'test' and not is_test):
                continue
            
            # 初始化此人的图像列表
            if subject_id not in self.subject_to_images:
                self.subject_to_images[subject_id] = []
            
            # 加载左右眼图像
            for eye in ['L', 'R']:
                eye_path = os.path.join(subject_path, eye)
                if os.path.exists(eye_path):
                    for img_name in os.listdir(eye_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            img_path = os.path.join(eye_path, img_name)
                            self.images.append(img_path)
                            self.subject_ids.append(subject_id)
                            self.subject_to_images[subject_id].append(img_path)
    
    def _load_thousand_dataset(self, thousand_path):
        """加载CASIA-Iris-Thousand数据集"""
        if not os.path.exists(thousand_path):
            print(f"路径不存在: {thousand_path}")
            return
            
        for subject_id in os.listdir(thousand_path):
            subject_path = os.path.join(thousand_path, subject_id)
            
            # 检查是否为文件夹
            if not os.path.isdir(subject_path):
                continue
            
            # 检查是训练集还是测试集
            subject_id_num = int(subject_id)
            is_test = (0 <= subject_id_num <= 199)
            
            # 如果模式与数据集不匹配则跳过
            if (self.mode == 'train' and is_test) or (self.mode == 'test' and not is_test):
                continue
            
            # 初始化此人的图像列表
            if subject_id not in self.subject_to_images:
                self.subject_to_images[subject_id] = []
            
            # 加载左右眼图像
            for eye in ['L', 'R']:
                eye_path = os.path.join(subject_path, eye)
                if os.path.exists(eye_path):
                    for img_name in os.listdir(eye_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            img_path = os.path.join(eye_path, img_name)
                            self.images.append(img_path)
                            self.subject_ids.append(subject_id)
                            self.subject_to_images[subject_id].append(img_path)
    
    def _load_gaze_dataset(self, gaze_path):
        """加载Ganzin-J7EF-Gaze数据集"""
        if not os.path.exists(gaze_path):
            print(f"路径不存在: {gaze_path}")
            return
            
        for subject_id in os.listdir(gaze_path):
            subject_path = os.path.join(gaze_path, subject_id)
            
            # 检查是否为文件夹
            if not os.path.isdir(subject_path):
                continue
            
            # 初始化此人的图像列表
            if subject_id not in self.subject_to_images:
                self.subject_to_images[subject_id] = []
            
            # 加载左右眼图像
            for eye in ['L', 'R']:
                eye_path = os.path.join(subject_path, eye)
                if os.path.exists(eye_path):
                    for img_name in os.listdir(eye_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            # 检查是否为测试集
                            is_test = False
                            if eye == 'L' and 'view_3' in img_name:
                                is_test = True
                            elif eye == 'R' and 'view_2' in img_name:
                                is_test = True
                                
                            # 如果模式与数据集不匹配则跳过
                            if (self.mode == 'train' and is_test) or (self.mode == 'test' and not is_test):
                                continue
                                
                            img_path = os.path.join(eye_path, img_name)
                            self.images.append(img_path)
                            self.subject_ids.append(subject_id)
                            self.subject_to_images[subject_id].append(img_path)
    
    def __len__(self):
        # 我们可以生成的不同图像对数量
        # 这里简单设为图像总数，因为我们在__getitem__中会随机生成图像对
        return len(self.images)
    
    def __getitem__(self, idx):
        # 随机决定是生成相同人的图像对还是不同人的图像对
        same_person = random.random() > 0.5
        
        if same_person:
            # 选一个有至少两张照片的人
            valid_subjects = [subj for subj, imgs in self.subject_to_images.items() if len(imgs) >= 2]
            if not valid_subjects:
                # 如果没有人有多张照片，改为选择不同人
                same_person = False
            else:
                # 选择一个人
                subject_id = random.choice(valid_subjects)
                # 从这个人的图像中随机选择两张不同的图像
                img_paths = random.sample(self.subject_to_images[subject_id], 2)
        
        if not same_person:
            # 选择两个不同的人
            subject_ids = random.sample(list(self.subject_to_images.keys()), 2)
            # 从每个人中随机选择一张图像
            img_paths = [random.choice(self.subject_to_images[subject_ids[0]]),
                        random.choice(self.subject_to_images[subject_ids[1]])]
        
        # 加载并转换图像
        try:
            images = []
            for img_path in img_paths:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            
            # 创建标签: 1表示相同人，0表示不同人
            label = torch.tensor(1.0 if same_person else 0.0, dtype=torch.float)
            
            return images[0], images[1], label
        except Exception as e:
            print(f"Error loading images: {e}")
            # 递归调用，重新随机选择
            return self.__getitem__(random.randint(0, len(self) - 1))


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # 使用预训练的GoogLeNet作为特征提取器
        googlenet = models.googlenet(pretrained=True)
        # 移除最后的全连接层
        self.feature_extractor = nn.Sequential(*list(googlenet.children())[:-2])
        # 添加全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 添加全连接层来比较特征向量
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
    
    def forward_one(self, x):
        # 提取特征
        x = self.feature_extractor(x)
        # 全局池化
        x = self.avg_pool(x)
        # 展平
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x1, x2):
        # 获取两张图像的特征表示
        feat1 = self.forward_one(x1)
        feat2 = self.forward_one(x2)
        
        # 计算特征差异
        diff = torch.abs(feat1 - feat2)
        
        # 通过全连接层处理差异
        out = self.fc(diff)
        
        # 应用sigmoid激活函数，输出相似性得分（0-1之间）
        return torch.sigmoid(out)

class ContrastiveLoss(nn.Module):
    """
    对比损失函数，用于训练孪生网络
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output, label):
        # 对比损失
        # output是网络输出的相似度分数（0-1之间）
        # label是1表示相同人，0表示不同人
        
        # 计算欧氏距离
        euclidean_distance = 1 - output  # 转换相似度到距离
        
        # 对比损失公式
        # 对于相同类别，我们希望距离小；对于不同类别，我们希望距离大于margin
        loss_contrastive = (label * euclidean_distance**2) + \
                          ((1 - label) * torch.clamp(self.margin - euclidean_distance, min=0)**2)
        
        return loss_contrastive.mean()


def get_data_loaders(batch_size=32, val_split=0.1):
    """
    创建训练、验证和测试数据加载器
    
    参数:
        batch_size (int): 批次大小
        val_split (float): 验证集比例
    
    返回:
        train_loader, val_loader, test_loader
    """
    # 定义图像预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = IrisDataset(mode='train', transform=train_transform)
    
    # 分割训练集和验证集
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # 创建测试集
    test_dataset = IrisDataset(mode='test', transform=test_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader


import os

def train_siamese_network(learning_rate=0.0001, num_epochs=30, resume_path=None):
    """
    训练孪生网络用于虹膜验证，可从某个模型权重继续训练
    
    参数:
        learning_rate (float): 学习率
        num_epochs (int): 训练轮数
        resume_path (str): 如果提供，加载这个路径的模型并继续训练
    
    返回:
        训练好的模型
    """
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32, val_split=0.1)
    
    model = SiameseNetwork()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = ContrastiveLoss(margin=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    
    # 嘗試從resume_path載入模型
    best_val_loss = float('inf')
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        print(f"🔁 Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint)
        # 若你有儲存 optimizer 和 epoch 資訊，也可載入這些（目前僅載入 model）
    else:
        print("🆕 Training from scratch.")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(start_epoch, num_epochs):
        # === Training ===
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for img1, img2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # === Validation ===
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for img1, img2, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                outputs = model(img1, img2).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predictions = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), r'D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\model\googlenet_50epoch.pth')
            print(f"💾 Model saved with validation loss: {best_val_loss:.4f}")



def verify_iris_images(model, img_path1, img_path2, threshold=0.8):
    """
    使用训练好的模型验证两张虹膜图像是否属于同一个人
    
    参数:
        model: 训练好的孪生网络模型
        img_path1: 第一张图像的路径
        img_path2: 第二张图像的路径
        threshold: 相似度阈值，高于此值则判定为同一个人
    
    返回:
        是否为同一个人的布尔值，相似度分数
    """
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    try:
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')
    except Exception as e:
        print(f"Error loading images: {e}")
        return False, 0.0
    
    # 应用转换
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)
    
    # 获取相似度得分
    with torch.no_grad():
        similarity = model(img1, img2).item()
    if similarity > threshold:
        return 1
    else:
        return 0



if __name__ == "__main__":
    # 检测可用的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 训练孪生网络
    # model = train_siamese_network(num_epochs=40)
    model = train_siamese_network(num_epochs=10,resume_path=r'D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\model\googlenet_40epoch.pth')
    
    # 示例：验证两张图像
    # 你可以替换为实际的图像路径
    # 加载训练好的模型
    # model = SiameseNetwork()
    # model.load_state_dict(torch.load('best_siamese_iris.pth'))
    
    # 示例验证
    # img_path1 = "path/to/image1.jpg"
    # img_path2 = "path/to/image2.jpg"
    # is_same, score = verify_iris_images(model, img_path1, img_path2)
    # print(f"Images belong to the same person: {is_same}")
    # print(f"Similarity score: {score:.4f}")