import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

def calculate_metrics_from_cm(cm, class_names=None):
    """
    混同行列からPrecision, Recall, F1-scoreを計算して表示する関数
    (classification_reportのような出力を生成)
    """
    # 各クラスごとのTP, FP, FN, TNを計算
    # TP: 対角成分
    true_positives = np.diag(cm)
    
    # FP: 列の合計 - TP
    false_positives = np.sum(cm, axis=0) - true_positives
    
    # FN: 行の合計 - TP
    false_negatives = np.sum(cm, axis=1) - true_positives
    
    # 精度の計算 (0除算回避のためにepsilonを加えるか、np.errstateを使う)
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    # NaNの処理（該当クラスの予測/正解が0個だった場合など）
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score)
    
    # サポート数（各クラスの正解ピクセル数）
    support = np.sum(cm, axis=1)

    # 結果をDataFrameにまとめる
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
        
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1_score,
        'Support': support
    }, index=class_names)
    
    # Accuracyの計算
    accuracy = np.sum(true_positives) / np.sum(cm)
    print(f"\nOverall Accuracy: {accuracy:.4f}\n")
    
    return metrics_df

# --- メイン処理 ---

# 設定
num_classes = 5  # クラス数（背景含む）
class_names = ["Background", "Class A", "Class B", "Class C", "Class D"]
total_cm = np.zeros((num_classes, num_classes), dtype=int)

# データローダーなどを想定したループ
# image_paths, label_paths は1000枚分のファイルパスリストと仮定
print("Processing images...")

for i in range(1000): # 1000枚の画像をループ
    # 1. 画像と正解ラベルを読み込む（ここはご自身のロード処理に置き換えてください）
    # ダミーデータ生成（例：512x512画像）
    y_true_img = np.random.randint(0, num_classes, (512, 512))
    y_pred_img = np.random.randint(0, num_classes, (512, 512))
    
    # 2. 1次元にフラット化 (この一瞬だけメモリを使うが、512x512なら問題ない)
    y_true_flat = y_true_img.flatten()
    y_pred_flat = y_pred_img.flatten()
    
    # 3. 混同行列を計算して累積する
    # labels引数でクラスIDを明示するのが重要（バッチ内に全クラスが含まれない場合があるため）
    cm_batch = confusion_matrix(y_true_flat, y_pred_flat, labels=range(num_classes))
    
    # 合計用の行列に足し込む
    total_cm += cm_batch

    # 進捗表示（任意）
    if (i + 1) % 100 == 0:
        print(f"{i + 1} images processed.")

# --- 結果の出力 ---
print("\n--- Aggregated Confusion Matrix ---")
print(total_cm)

print("\n--- Classification Report (Calculated manually) ---")
report_df = calculate_metrics_from_cm(total_cm, class_names)
print(report_df)

# IoUの計算
intersection = np.diag(total_cm)
union = np.sum(total_cm, axis=0) + np.sum(total_cm, axis=1) - intersection
iou = intersection / union
mean_iou = np.nanmean(iou)

print("\n--- IoU Scores ---")
for i, name in enumerate(class_names):
    print(f"{name}: {iou[i]:.4f}")
print(f"Mean IoU: {mean_iou:.4f}")
