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
import re
from PIL import Image
from PIL.TiffTags import TAGS

def calculate_lat_force_max(file_path):
    """
    GeoTIFFタグ内の全数値から、Y座標（Northing）と思われる最大値を探索して計算する
    """
    try:
        with Image.open(file_path) as img:
            tags = img.tag_v2
            
            # --- 1. ModelTiepointTag (33922) の全データを取得 ---
            if 33922 not in tags:
                return None, "No ModelTiepointTag found"
            
            tiepoints = tags[33922]
            
            # デバッグ用: 取得した生データを文字列にする
            raw_data_str = str(tiepoints)
            
            # --- 2. Y座標（Northing）の特定 ---
            # UTM座標系では、緯度10度以上の地域において、
            # Y座標(数百万m) は X座標(数十万m) や ピクセル座標(0や1000) より圧倒的に大きいです。
            # したがって、タプルの中にある「最大の数値」を探せば、それがY座標です。
            
            # 念のため 0.0 などを除外したリストを作る（なくても動きますが安全のため）
            candidates = [val for val in tiepoints if isinstance(val, (int, float))]
            
            if not candidates:
                return None, "ModelTiepointTag is empty"
                
            y_coordinate = max(candidates)
            
            # --- 3. 半球の判定 ---
            is_southern_hemisphere = False
            geo_ascii = ""
            if 34737 in tags:
                geo_ascii_raw = tags[34737]
                if isinstance(geo_ascii_raw, bytes):
                    geo_ascii = geo_ascii_raw.decode('ascii', errors='ignore')
                else:
                    geo_ascii = str(geo_ascii_raw)
                
                # 南半球判定
                if re.search(r'zone\s*\d+S', geo_ascii, re.IGNORECASE) or "southern" in geo_ascii.lower():
                    is_southern_hemisphere = True
            
            # --- 4. 緯度の計算 ---
            meters_per_degree = 111111.0
            
            if is_southern_hemisphere:
                # 南半球 (UTMのY座標は 10,000,000 から減っていく)
                # 例: 南緯35度なら Y座標は約600万。1000万-600万=400万(距離)。
                distance_from_equator = 10000000.0 - y_coordinate
                latitude = -1 * (distance_from_equator / meters_per_degree)
            else:
                # 北半球
                latitude = y_coordinate / meters_per_degree
                
            # デバッグ情報を返す
            debug_info = (f"RawTags={raw_data_str}, "
                          f"SelectedY={y_coordinate}, "
                          f"Hemisphere={'South' if is_southern_hemisphere else 'North'}")
            
            return latitude, debug_info

    except Exception as e:
        return None, str(e)
def calculate_lat_pil(file_path):
    """
    GeoTIFFのメタ情報から緯度を計算する（軸順序の入れ替わりに対応した堅牢版）
    """
    ity_dirs = [p for p in glob.glob(os.path.join(dataset_root, "*")) if os.path.isdir(p)]
    
    toropical=[]
    
    for city_full_path in city_dirs:
        # パスから都市名だけを取り出す (例: "aachen")
        city_name = os.path.basename(city_full_path)
        
        # 4. 各都市の中にある "images" フォルダのパスを作成
        images_dir = os.path.join(city_full_path, "images")
        
        # 5. その中の .tif ファイルを取得
        tif_files = glob.glob(os.path.join(images_dir, "*.tif"))
        tif_file = None
        if tif_files is not None:
            tif_file = tif_files[:1]
        else:
            continue
        
        # 確認用出力
        print(f"=== 都市: {city_name} (ファイル数: {len(tif_file)}) ===")
        for path in tif_file:
            is_subtorpical_area = is_subtropical(path)
            if is_subtorpical_area:
                toropical.append(city_name)
    try:
        with Image.open(file_path) as img:
            tags = img.tag_v2
            
            # --- 1. ModelTiepointTag (33922) の取得 ---
            if 33922 not in tags:
                return None, "No ModelTiepointTag"
            
            tiepoints = tags[33922]
            
            # 要素数が足りない場合はエラー
            if len(tiepoints) < 6:
                return None, "Invalid ModelTiepointTag format"
            
            # 候補となる2つの値を取得 (通常は [3]がX, [4]がY だが、逆の場合がある)
            val1 = tiepoints[3]
            val2 = tiepoints[4]
            
            # --- 2. Y座標（Northing）の推定 ---
            # 中緯度地域（日本、欧州など）では、Y座標（赤道からの距離）は X座標（ゾーン内の横位置）より
            # 桁違いに大きい値（数百万メートル）になります。
            # そのため、大きい方をY座標として採用します。
            # ※赤道直下(緯度4度未満)ではXの方が大きくなる可能性がありますが、
            #   その場合でも計算結果は「亜熱帯」となるため、判定目的では問題ありません。
            
            y_coordinate = max(val1, val2)
            
            # 参考: どちらが選ばれたかを確認するためのログ用
            used_index = 3 if y_coordinate == val1 else 4

            # --- 3. 半球の判定 ---
            is_southern_hemisphere = False
            if 34737 in tags:
                geo_ascii = tags[34737]
                if isinstance(geo_ascii, bytes):
                    geo_ascii = geo_ascii.decode('ascii', errors='ignore')
                
                # "Zone 54S" や "Southern Hemisphere" を検出
                if re.search(r'zone\s*\d+S', geo_ascii, re.IGNORECASE) or "southern" in geo_ascii.lower():
                    is_southern_hemisphere = True
            
            # --- 4. 緯度の計算 ---
            meters_per_degree = 111111.0
            
            if is_southern_hemisphere:
                # 南半球: 10,000,000m - Y座標
                distance_from_equator = 10000000.0 - y_coordinate
                latitude = -1 * (distance_from_equator / meters_per_degree)
            else:
                # 北半球: Y座標そのまま
                latitude = y_coordinate / meters_per_degree
                
            return latitude, f"Hemisphere: {'South' if is_southern_hemisphere else 'North'}, UsedIndex: {used_index}"

    except Exception as e:
        return None, str(e)
# ==========================================
# 判定ロジック（亜熱帯判定）
# ==========================================
def is_subtropical(file_path):
    lat, info = calculate_lat_pil(file_path)
    
    if lat is None:
        return f"Error: {info}"
    
    # 判定基準: 南北緯30度以内なら亜熱帯・熱帯とする
    # abs()で絶対値をとることで、南緯20度(-20)も20として判定
    if abs(lat) <= 30.0:
        print(f"亜熱帯です (緯度: {lat:.2f}, {info})")
        return True
    else:
        print(f"亜熱帯ではありません (緯度: {lat:.2f}, {info})")
        return False
