import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import seaborn as sns
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

class SiameseVisualizationToolkit:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # 預處理設定
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 反正規化設定（用於可視化）
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
    
    def preprocess_image(self, image_path):
        """預處理圖像"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return image, image_tensor
    
    def method1_gradient_based_attention(self, image1_tensor, image2_tensor):
        """方法1: 基於梯度的注意力可視化"""
        print("🔍 使用方法1: 基於梯度的注意力可視化")
        
        # 啟用梯度計算
        image1_tensor.requires_grad_(True)
        image2_tensor.requires_grad_(True)
        
        # 前向傳播
        similarity = self.model(image1_tensor, image2_tensor)
        
        # 反向傳播
        self.model.zero_grad()
        similarity.backward()
        
        # 獲取輸入梯度
        grad1 = image1_tensor.grad.data
        grad2 = image2_tensor.grad.data
        
        # 計算注意力圖 - 使用梯度的絕對值和通道平均
        attention1 = torch.abs(grad1).mean(dim=1).squeeze().cpu().numpy()
        attention2 = torch.abs(grad2).mean(dim=1).squeeze().cpu().numpy()
        
        # 正規化到0-1
        attention1 = self.normalize_attention_map(attention1)
        attention2 = self.normalize_attention_map(attention2)
        
        return attention1, attention2, similarity.item()
    
    def method2_integrated_gradients(self, image1_tensor, image2_tensor, steps=50):
        """方法2: 積分梯度法 (Integrated Gradients)"""
        print("🔍 使用方法2: 積分梯度法")
        
        def get_gradients(img1, img2):
            img1.requires_grad_(True)
            img2.requires_grad_(True)
            
            similarity = self.model(img1, img2)
            self.model.zero_grad()
            similarity.backward()
            
            return img1.grad.data, img2.grad.data, similarity.item()
        
        # 創建基線（零圖像）
        baseline1 = torch.zeros_like(image1_tensor)
        baseline2 = torch.zeros_like(image2_tensor)
        
        # 積分梯度計算
        integrated_grads1 = torch.zeros_like(image1_tensor)
        integrated_grads2 = torch.zeros_like(image2_tensor)
        
        for i in range(steps):
            # 線性插值
            alpha = i / steps
            interpolated1 = baseline1 + alpha * (image1_tensor - baseline1)
            interpolated2 = baseline2 + alpha * (image2_tensor - baseline2)
            
            # 計算梯度
            grad1, grad2, _ = get_gradients(interpolated1, interpolated2)
            
            integrated_grads1 += grad1
            integrated_grads2 += grad2
        
        # 平均並乘以輸入差異
        integrated_grads1 = integrated_grads1 * (image1_tensor - baseline1) / steps
        integrated_grads2 = integrated_grads2 * (image2_tensor - baseline2) / steps
        
        # 轉換為注意力圖
        attention1 = torch.abs(integrated_grads1).mean(dim=1).squeeze().cpu().numpy()
        attention2 = torch.abs(integrated_grads2).mean(dim=1).squeeze().cpu().numpy()
        
        attention1 = self.normalize_attention_map(attention1)
        attention2 = self.normalize_attention_map(attention2)
        
        # 獲取最終相似性分數
        with torch.no_grad():
            similarity = self.model(image1_tensor, image2_tensor).item()
        
        return attention1, attention2, similarity
    
    def method3_feature_difference_map(self, image1_tensor, image2_tensor):
        """方法3: 特徵差異圖"""
        print("🔍 使用方法3: 特徵差異圖")
        
        # 提取中間層特徵
        def extract_features(x):
            features = []
            for i, layer in enumerate(self.model.feature_extractor):
                x = layer(x)
                # 收集卷積層的輸出
                if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)) and len(x.shape) == 4:
                    features.append(x)
            return features, x
        
        with torch.no_grad():
            features1, final1 = extract_features(image1_tensor)
            features2, final2 = extract_features(image2_tensor)
            
            similarity = self.model(image1_tensor, image2_tensor).item()
        
        # 計算特徵差異
        attention_maps = []
        for i, (f1, f2) in enumerate(zip(features1, features2)):
            # 計算特徵差異
            diff = torch.abs(f1 - f2)
            # 通道平均
            diff_map = diff.mean(dim=1).squeeze().cpu().numpy()
            # 調整大小到輸入尺寸
            diff_map = cv2.resize(diff_map, (224, 224))
            attention_maps.append(self.normalize_attention_map(diff_map))
        
        # 使用最後幾層的平均
        if len(attention_maps) >= 2:
            attention1 = np.mean(attention_maps[-2:], axis=0)
            attention2 = attention1.copy()  # 差異圖對兩張圖像是相同的
        else:
            attention1 = attention_maps[-1] if attention_maps else np.ones((224, 224)) * 0.5
            attention2 = attention1.copy()
        
        return attention1, attention2, similarity
    
    def method4_guided_backprop(self, image1_tensor, image2_tensor):
        """方法4: 引導反向傳播 (Guided Backpropagation)"""
        print("🔍 使用方法4: 引導反向傳播")
        
        # 修改ReLU的反向傳播
        def relu_hook_function(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        
        # 註冊鉤子
        hooks = []
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_backward_hook(relu_hook_function))
        
        try:
            image1_tensor.requires_grad_(True)
            image2_tensor.requires_grad_(True)
            
            similarity = self.model(image1_tensor, image2_tensor)
            self.model.zero_grad()
            similarity.backward()
            
            grad1 = image1_tensor.grad.data
            grad2 = image2_tensor.grad.data
            
            attention1 = torch.abs(grad1).mean(dim=1).squeeze().cpu().numpy()
            attention2 = torch.abs(grad2).mean(dim=1).squeeze().cpu().numpy()
            
            attention1 = self.normalize_attention_map(attention1)
            attention2 = self.normalize_attention_map(attention2)
            
        finally:
            # 移除鉤子
            for hook in hooks:
                hook.remove()
        
        return attention1, attention2, similarity.item()
    
    def method5_contrastive_attention(self, image1_tensor, image2_tensor):
        """方法5: 對比注意力 - Siamese網路專用"""
        print("🔍 使用方法5: 對比注意力（Siamese專用）")
        
        with torch.no_grad():
            # 提取特徵
            feat1 = self.model.forward_one(image1_tensor)
            feat2 = self.model.forward_one(image2_tensor)
            
            # 計算相似性
            similarity = self.model(image1_tensor, image2_tensor).item()
        
        # 創建擾動版本來測試敏感性
        attention1 = self.compute_sensitivity_map(image1_tensor, image2_tensor, target='img1')
        attention2 = self.compute_sensitivity_map(image1_tensor, image2_tensor, target='img2')
        
        return attention1, attention2, similarity
    
    def compute_sensitivity_map(self, img1, img2, target='img1', noise_level=0.1, num_samples=20):
        """計算敏感性圖"""
        original_sim = self.model(img1, img2).item()
        
        if target == 'img1':
            target_img = img1
        else:
            target_img = img2
        
        sensitivity_map = torch.zeros((224, 224))
        
        # 滑動窗口遮擋
        window_size = 16
        stride = 8
        
        for i in range(0, 224 - window_size, stride):
            for j in range(0, 224 - window_size, stride):
                # 創建遮擋版本
                masked_img = target_img.clone()
                masked_img[:, :, i:i+window_size, j:j+window_size] = 0
                
                # 計算相似性變化
                if target == 'img1':
                    new_sim = self.model(masked_img, img2).item()
                else:
                    new_sim = self.model(img1, masked_img).item()
                
                # 敏感性 = 相似性變化的絕對值
                sensitivity = abs(original_sim - new_sim)
                sensitivity_map[i:i+window_size, j:j+window_size] = max(
                    sensitivity_map[i:i+window_size, j:j+window_size].max(), 
                    sensitivity
                )
        
        return self.normalize_attention_map(sensitivity_map.numpy())
    
    def normalize_attention_map(self, attention_map):
        """正規化注意力圖到0-1範圍"""
        if attention_map.max() > attention_map.min():
            return (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        else:
            return np.ones_like(attention_map) * 0.5
    
    def comprehensive_siamese_visualization(self, image1_path, image2_path, save_dir=None):
        """綜合Siamese網路可視化"""
        print("=" * 60)
        print("🎯 Siamese網路專用綜合可視化分析")
        print("=" * 60)
        
        # 載入圖像
        image1, tensor1 = self.preprocess_image(image1_path)
        image2, tensor2 = self.preprocess_image(image2_path)
        
        # 所有方法
        methods = {
            '梯度注意力': self.method1_gradient_based_attention,
            '積分梯度': self.method2_integrated_gradients,
            '特徵差異': self.method3_feature_difference_map,
            '引導反向傳播': self.method4_guided_backprop,
            '對比注意力': self.method5_contrastive_attention
        }
        
        results = {}
        
        # 嘗試所有方法
        for method_name, method_func in methods.items():
            try:
                print(f"\n🔄 執行 {method_name}...")
                attention1, attention2, similarity = method_func(tensor1.clone(), tensor2.clone())
                
                # 檢查結果有效性
                if (attention1.std() > 1e-6 and attention2.std() > 1e-6):
                    results[method_name] = {
                        'attention1': attention1,
                        'attention2': attention2,
                        'similarity': similarity,
                        'status': 'success'
                    }
                    print(f"✅ {method_name} 成功")
                    print(f"   相似性: {similarity:.4f}")
                    print(f"   注意力圖1變異: {attention1.std():.6f}")
                    print(f"   注意力圖2變異: {attention2.std():.6f}")
                else:
                    results[method_name] = {'status': 'failed', 'reason': '注意力圖無變化'}
                    print(f"❌ {method_name} 失敗: 注意力圖無變化")
                    
            except Exception as e:
                results[method_name] = {'status': 'failed', 'reason': str(e)}
                print(f"❌ {method_name} 失敗: {e}")
        
        # 找到最佳方法
        successful_methods = {k: v for k, v in results.items() if v['status'] == 'success'}
        
        if not successful_methods:
            print("❌ 所有方法都失敗了！")
            return results
        
        print(f"\n✅ 成功的方法: {list(successful_methods.keys())}")
        
        # 可視化所有成功的方法
        self.plot_comparison(image1, image2, successful_methods, save_dir)
        
        # 詳細分析
        self.detailed_analysis(successful_methods)
        
        return results
    
    def plot_comparison(self, image1, image2, results, save_dir=None):
        """繪製比較圖"""
        num_methods = len(results)
        fig, axes = plt.subplots(num_methods, 5, figsize=(22, 5*num_methods))
        
        # 調整子圖間距
        plt.subplots_adjust( hspace=0.5)

        if num_methods == 1:
            axes = axes.reshape(1, -1)

        for i, (method_name, data) in enumerate(results.items()):
            attention1 = data['attention1']
            attention2 = data['attention2']
            similarity = data['similarity']

            # 原始圖像
            axes[i, 0].imshow(image1)
            axes[i, 0].set_title(f'Image 1\n({method_name})', pad=10, fontsize=12)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(image2)
            axes[i, 1].set_title(f'Image 2\n(Sim: {similarity:.4f})', pad=10, fontsize=12)
            axes[i, 1].axis('off')

            # 注意力圖
            im1 = axes[i, 2].imshow(attention1, cmap='jet', vmin=0, vmax=1)
            axes[i, 2].set_title('Attention Map 1', pad=10, fontsize=12)
            axes[i, 2].axis('off')
            plt.colorbar(im1, ax=axes[i, 2], fraction=0.046, pad=0.04)

            im2 = axes[i, 3].imshow(attention2, cmap='jet', vmin=0, vmax=1)
            axes[i, 3].set_title('Attention Map 2', pad=10, fontsize=12)
            axes[i, 3].axis('off')
            plt.colorbar(im2, ax=axes[i, 3], fraction=0.046, pad=0.04)

            # 差異圖
            diff_map = np.abs(attention1 - attention2)
            im3 = axes[i, 4].imshow(diff_map, cmap='hot', vmin=0, vmax=1)
            axes[i, 4].set_title('Attention Difference', pad=10, fontsize=12)
            axes[i, 4].axis('off')
            plt.colorbar(im3, ax=axes[i, 4], fraction=0.046, pad=0.04)

        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/siamese_comprehensive_analysis.png", 
                        dpi=300, bbox_inches='tight')

        plt.show()

    
    def detailed_analysis(self, results):
        """詳細分析結果"""
        print("\n" + "=" * 50)
        print("📊 詳細分析結果")
        print("=" * 50)
        
        for method_name, data in results.items():
            attention1 = data['attention1']
            attention2 = data['attention2']
            similarity = data['similarity']
            
            print(f"\n🔹 {method_name}:")
            print(f"   相似性分數: {similarity:.6f}")
            print(f"   注意力圖1統計: mean={attention1.mean():.4f}, std={attention1.std():.4f}")
            print(f"   注意力圖2統計: mean={attention2.mean():.4f}, std={attention2.std():.4f}")
            print(f"   注意力差異: {np.abs(attention1 - attention2).mean():.4f}")
            
            # 找到最高注意力區域
            max_pos1 = np.unravel_index(attention1.argmax(), attention1.shape)
            max_pos2 = np.unravel_index(attention2.argmax(), attention2.shape)
            print(f"   最高注意力位置1: {max_pos1}")
            print(f"   最高注意力位置2: {max_pos2}")
    
    def quick_test(self, image1_path, image2_path):
        """快速測試 - 推薦最適合的方法"""
        print("🚀 快速測試模式")
        
        image1, tensor1 = self.preprocess_image(image1_path)
        image2, tensor2 = self.preprocess_image(image2_path)
        
        # 先嘗試最可靠的方法
        try:
            print("嘗試對比注意力方法...")
            attention1, attention2, similarity = self.method5_contrastive_attention(
                tensor1.clone(), tensor2.clone())
            
            if attention1.std() > 1e-6:
                print(f"✅ 成功！相似性: {similarity:.4f}")
                self.plot_single_result(image1, image2, attention1, attention2, 
                                      similarity, "對比注意力")
                return attention1, attention2, similarity
        except Exception as e:
            print(f"❌ 對比注意力失敗: {e}")
        
        # 備用方案
        try:
            print("嘗試梯度注意力方法...")
            attention1, attention2, similarity = self.method1_gradient_based_attention(
                tensor1.clone(), tensor2.clone())
            
            if attention1.std() > 1e-6:
                print(f"✅ 成功！相似性: {similarity:.4f}")
                self.plot_single_result(image1, image2, attention1, attention2, 
                                      similarity, "梯度注意力")
                return attention1, attention2, similarity
        except Exception as e:
            print(f"❌ 梯度注意力失敗: {e}")
        
        print("❌ 所有快速方法都失敗了，建議使用comprehensive_siamese_visualization進行完整分析")
        return None, None, None
    
    def plot_single_result(self, image1, image2, attention1, attention2, similarity, method_name):
        """繪製單個結果"""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(image1)
        axes[0].set_title('Image 1')
        axes[0].axis('off')
        
        axes[1].imshow(image2)
        axes[1].set_title('Image 2')
        axes[1].axis('off')
        
        im1 = axes[2].imshow(attention1, cmap='jet')
        axes[2].set_title('Attention 1')
        axes[2].axis('off')
        plt.colorbar(im1, ax=axes[2])
        
        im2 = axes[3].imshow(attention2, cmap='jet')
        axes[3].set_title('Attention 2')
        axes[3].axis('off')
        plt.colorbar(im2, ax=axes[3])
        
        plt.suptitle(f'{method_name} - Similarity: {similarity:.4f}', fontsize=14)
        plt.tight_layout()
        plt.show()

# 使用範例
def test_siamese_visualization(model_path, image1_path, image2_path, device='cpu'):
    """測試Siamese可視化工具"""
    import os
    
    # 載入模型（根據你的SiameseNetwork類別）
    from tune_googlenet import SiameseNetwork  # 請替換為實際的導入
    
    model = SiameseNetwork()
    
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("✅ 模型載入成功")
    else:
        print("⚠️  使用未訓練的模型")
    
    model.to(device)
    
    # 創建可視化工具
    viz_tool = SiameseVisualizationToolkit(model, device)
    
    # 快速測試
    # print("=" * 60)
    # print("🚀 快速測試")
    # viz_tool.quick_test(image1_path, image2_path)
    
    # 綜合分析
    print("\n" + "=" * 60)
    print("🎯 綜合分析")
    results = viz_tool.comprehensive_siamese_visualization(
        image1_path, image2_path, save_dir='siamese_visualization_results')
    
    return viz_tool, results

# 直接運行範例
if __name__ == "__main__":
    # 設定參數
    MODEL_PATH = r"D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\model\googlenet_50epoch.pth"
    IMAGE1_PATH = r"D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\dataset\CASIA-Iris-Thousand\015\L\S5015L08.jpg"
    IMAGE2_PATH = r"D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\dataset\CASIA-Iris-Thousand\015\L\S5015L00.jpg"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 執行測試
    viz_tool, results = test_siamese_visualization(MODEL_PATH, IMAGE1_PATH, IMAGE2_PATH, DEVICE)