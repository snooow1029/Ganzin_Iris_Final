import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_d_prime(genuine_scores, imposter_scores):
    genuine_scores = np.array(genuine_scores)
    imposter_scores = np.array(imposter_scores)

    g_mean = np.mean(genuine_scores)
    g_var = np.var(genuine_scores)
    i_mean = np.mean(imposter_scores)
    i_var = np.var(imposter_scores)
    d_prime = np.absolute(g_mean - i_mean) / np.sqrt(0.5 * (g_var + i_var))

    return d_prime

def get_uid(file_path):
    p = Path(file_path)
    img_name = p.parts[-1]
    eye = p.parts[-2]
    uid = p.parts[-3]
    dataset = p.parts[-4]
    # print(f"dataset: {dataset}, uid: {uid}, eye: {eye}, img_name: {img_name}")

    return uid

import os
def plot_score_distribution(genuine_scores, imposter_scores, output_path='score_distribution.png'):
    plt.figure(figsize=(8, 5))
    plt.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine', color='blue', density=True)
    plt.hist(imposter_scores, bins=50, alpha=0.5, label='Imposter', color='red', density=True)

    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title("Score Distribution (Genuine vs Imposter)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"[Info] Score distribution plot saved to {output_path}")
    plt.close()
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_f1(genuine_scores, imposter_scores, threshold=0.5):
    """
    將分數高於 threshold 判定為 genuine
    """
    y_true = [1] * len(genuine_scores) + [0] * len(imposter_scores)
    y_pred = [1 if score >= threshold else 0 for score in genuine_scores + imposter_scores]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Iris Recognition Evaluation')
    parser.add_argument('--input', type=str, metavar='PATH', required=True,
                        help='A list of sampled pairs and the results')
    parser.add_argument('--save_plot', type=str, default=None,
                        help='Output path to save the score distribution plot')

    args = parser.parse_args()

    # 根據 input 自動產生 save_plot 路徑
    if args.save_plot is None:
        base, ext = os.path.splitext(args.input)
        args.save_plot = f"{base}_distribution.png"

    genuine_scores = []
    imposter_scores = []

    with open(args.input, 'r') as file:
        for line in file:
            lineparts = line.split(',')
            score = float(lineparts[2].strip())

            if score < 0 or score > 1:
                print("[Error] score should be normalized to 0~1 before evaluation")
                print(line)
                exit(1)

            id1 = get_uid(lineparts[0].strip())
            id2 = get_uid(lineparts[1].strip())

            if id1 == id2:
                genuine_scores.append(score)
            else:
                imposter_scores.append(score)

    d_prime = calculate_d_prime(genuine_scores, imposter_scores)
    print(f"d' score = {d_prime:.4f}")
    precision, recall, f1 = calculate_f1(genuine_scores, imposter_scores, threshold=0.7)
    print(f"Precision = {precision:.4f}")
    print(f"Recall    = {recall:.4f}")
    print(f"F1 Score  = {f1:.4f}")
    plot_score_distribution(genuine_scores, imposter_scores, output_path=args.save_plot)

