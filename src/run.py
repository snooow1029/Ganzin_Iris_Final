import argparse
import numpy as np
import cv2
from algorithm import compute_orb_similarity
from tune_googlenet import verify_iris_images, SiameseNetwork
#from tune_googlenet_pre import verify_iris_images_with_preprocessing,SiameseNetwork
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ganzin Iris Recognition Challenge')
    parser.add_argument('--input', type=str, metavar='PATH', required=True,
                        help='Input file to specify a list of sampled pairs')
    parser.add_argument('--output', type=str, metavar='PATH', required=True,
                        help='A list of sampled pairs and the testing results')

    args = parser.parse_args()

    model = SiameseNetwork()
    # model = SiameseNetworkWithSTN()
    
    # 将模型移至GPU（如果可用）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(r'D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\model\googlenet_posttrain.pth'))
    # checkpoint = torch.load(r'D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\model\googlenet_posttrain.pth', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # 在测试集上评估模型
    model.eval()
    # img1_path = r"D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\dataset\CASIA-Iris-Lamp\002\L\S2002L03.jpg"     # 替換為第一張圖片路徑
    # img2_path = r"D:\家愷的資料\大學\大三\電腦視覺\final\Ganzin_supplement4student\dataset\CASIA-Iris-Lamp\007\R\S2007R03.jpg"     # 替換為第二張圖片路徑
    # score = verify_iris_images_with_preprocessing(model, img1_path, img2_path,threshold=0.85)
    # output_line = f"{img1_path}, {img2_path}, {score}"


    with open(args.input, 'r') as in_file, open(args.output, 'w') as out_file:
        for line in in_file:
            lineparts = line.split(',')
            img1_path = lineparts[0].strip()
            img2_path = lineparts[1].strip()

            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            # TODO: Replace with your algorithm
            # score = compute_orb_similarity(img1, img2)
            score = verify_iris_images(model, img1_path, img2_path,threshold=0.9)
            # score = verify_iris_images_with_preprocessing(model, img1_path, img2_path,threshold=0.85)
            output_line = f"{img1_path}, {img2_path}, {score}"
            # print(output_line)
            out_file.write(output_line.rstrip('\n') + '\n')
