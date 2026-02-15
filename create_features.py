import os
import numpy as np
from skimage import color, filters, morphology
from skimage.filters import rank
from scipy import ndimage

def _create_feature_stack(image):
    img_float = image.astype(np.float32) / 255.0
    
    # グレースケール
    gray = color.rgb2gray(img_float)
    gray_uint8 = (gray * 255).astype(np.uint8)
    
    features = []
    
    # RGB
    features.append(img_float)
    
    # HSV 
    hsv = color.rgb2hsv(img_float)
    features.append(hsv)
    
    # edge 
    edge = filters.sobel(gray)
    features.append(edge[..., np.newaxis])
    
    """
    # Texture / Local Entropy
    radius = 3
    selem = morphology.disk(radius)
    entropy = rank.entropy(gray_uint8, selem)
    # 0-1に正規化
    features.append((entropy.astype(np.float32) / 255.0)[..., np.newaxis])
    """
    
    # --- Feature 4: Additional Indices (Optional) ---
    # 例: GLI (Green Leaf Index) - RGBだけで植生を強調
    # R, G, B = img_float[..., 0], img_float[..., 1], img_float[..., 2]
    # gli = (2 * G - R - B) / (2 * G + R + B + 1e-6)
    # features.append(gli[..., np.newaxis])
    
    # gaussian blue
    gray_sqr = gray ** 2
    scales=[1.0, 2.0, 4.0, 8.0]
    for sigma in scales:
        # A. Gaussian Blur (周辺平均色) - Context
        # (H, W, 3) -> 各チャンネルぼかす
        blurred = filters.gaussian(img_float, sigma=sigma, channel_axis=-1)
        features.append(blurred)

        # B. Local Std Dev (周辺分散) - Texture
        # 高速計算: Var(X) = E[X^2] - (E[X])^2
        # sigmaが大きいほど「広い範囲の均質性」を見る
        mean_gray = filters.gaussian(gray, sigma=sigma)
        sqr_mean_gray = filters.gaussian(gray_sqr, sigma=sigma)
        
        # 浮動小数点誤差対策 (負にならないように)
        std_gray = np.sqrt(np.maximum(sqr_mean_gray - mean_gray**2, 0))
        features.append(std_gray[..., np.newaxis])
    
    # Local std dev
    mean_gray = filters.gaussian(gray, sigma=sigma)
    sqr_mean_gray = filters.gaussian(gray**2, sigma=sigma)
    # 浮動小数点誤差対策でclip
    std_gray = np.sqrt(np.maximum(sqr_mean_gray - mean_gray**2, 0))
    features.append(std_gray[..., np.newaxis])

    # すべての特徴量を深さ方向に結合
    # 出力形状: (H, W, Total_Features)
    return np.dstack(features)

def extract_training_features(image, target_pixels, patch_size=33):
    """
    ランダムフォレスト学習用の特徴量テーブルを作成する。
    画像全体を処理せず、必要なピクセルの周囲だけを切り出して計算する（省メモリ）。

    Parameters
    ----------
    image : ndarray (H, W, 3), uint8
    target_pixels : tuple (y_indices, x_indices)
    patch_size : int (奇数推奨, 特徴量計算に必要な受容野サイズ)

    Returns
    -------
    X_train : ndarray (N_samples, N_features)
    """
    y_idx, x_idx = target_pixels
    num_samples = len(y_idx)
    
    # 1つのサンプルで特徴量の次元数を確認するためにダミー実行
    dummy_patch = np.zeros((patch_size, patch_size, 3), dtype=image.dtype)
    dummy_feat = _create_feature_stack(dummy_patch)
    n_features = dummy_feat.shape[2]
    
    # 結果格納用配列
    X_train = np.zeros((num_samples, n_features), dtype=np.float32)
    
    radius = patch_size // 2
    H, W = image.shape[:2]

    # パディング（画像の端のピクセルが指定された場合用）
    # reflectモードで鏡像パディングしておくと自然なテクスチャ計算が可能
    pad_width = ((radius, radius), (radius, radius), (0, 0))
    image_padded = np.pad(image, pad_width, mode='reflect')

    # 座標もパディング分ずらす
    y_idx_pad = y_idx + radius
    x_idx_pad = x_idx + radius

    for i in range(num_samples):
        py, px = y_idx_pad[i], x_idx_pad[i]
        
        # パッチ切り出し
        # パディング済み画像から切り出すので、境界チェック不要
        patch = image_padded[py-radius : py+radius+1, px-radius : px+radius+1]
        
        # 特徴量計算 (H_patch, W_patch, n_feats)
        feat_map = _create_feature_stack(patch)
        
        # パッチの中心（=ターゲットピクセル）の特徴量を取得
        # 中心座標は (radius, radius)
        X_train[i, :] = feat_map[radius, radius, :]

    return X_train
    
def predict_large_image(model, image, tile_size=512, overlap=32):
    """
    巨大画像に対して、学習済みモデルで推論を行う。
    タイル分割＋オーバーラップ処理で、メモリ不足と境界ノイズを防ぐ。

    Parameters
    ----------
    model : sklearn estimator (RandomForestClassifier etc.)
    image : ndarray (H, W, 3), uint8
    tile_size : int (一度に処理するサイズ)
    overlap : int (のりしろサイズ。フィルタ半径より大きくすること)

    Returns
    -------
    prediction_map : ndarray (H, W), uint8 (クラスID)
    """
    h, w, _ = image.shape
    
    # 出力画像
    prediction_map = np.zeros((h, w), dtype=np.uint8)
    
    # ループ処理 (Y方向)
    for y in range(0, h, tile_size):
        # 今回のタイルの有効範囲 (Y)
        y_valid_start = y
        y_valid_end = min(y + tile_size, h)
        h_valid = y_valid_end - y_valid_start
        
        # タイル切り出し範囲（オーバーラップ込み）
        y_cut_start = max(0, y - overlap)
        y_cut_end = min(h, y + tile_size + overlap)
        
        # ループ処理 (X方向)
        for x in range(0, w, tile_size):
            # 今回のタイルの有効範囲 (X)
            x_valid_start = x
            x_valid_end = min(x + tile_size, w)
            w_valid = x_valid_end - x_valid_start
            
            # タイル切り出し範囲（オーバーラップ込み）
            x_cut_start = max(0, x - overlap)
            x_cut_end = min(w, x + tile_size + overlap)
            
            # 1. 画像切り出し
            tile = image[y_cut_start:y_cut_end, x_cut_start:x_cut_end]
            
            # 端っこでオーバーラップが足りない場合のパディング処理
            # (フィルタ処理でサイズが変わらないようにするため)
            pad_y = (overlap - (y_valid_start - y_cut_start), 
                     overlap - (y_cut_end - y_valid_end))
            pad_x = (overlap - (x_valid_start - x_cut_start), 
                     overlap - (x_cut_end - x_valid_end))
            
            if any(p > 0 for p in pad_y + pad_x):
                 tile = np.pad(tile, (pad_y, pad_x, (0,0)), mode='reflect')

            # 2. 特徴量計算 (ここが重いのでタイル一括処理)
            # tile shape: (H_tile, W_tile, 3) -> (H_tile, W_tile, n_feats)
            features_map = _create_feature_stack(tile)
            
            # 3. 推論用にReshape
            n_samples_tile = features_map.shape[0] * features_map.shape[1]
            n_features = features_map.shape[2]
            X_tile = features_map.reshape(n_samples_tile, n_features)
            
            # NaN対策 (0除算などで発生した場合)
            X_tile = np.nan_to_num(X_tile)
            
            # 4. 推論実行
            pred_tile_flat = model.predict(X_tile)
            pred_tile_map = pred_tile_flat.reshape(features_map.shape[0], features_map.shape[1])
            
            # 5. オーバーラップ（のりしろ）を除去して、有効領域だけを取り出す
            # パディングした分も考慮して、中心部分をくり抜く
            # ここでの offset は、pad処理後の tile 座標系における有効領域の開始位置
            offset_y = overlap
            offset_x = overlap
            
            valid_pred = pred_tile_map[offset_y : offset_y + h_valid, 
                                       offset_x : offset_x + w_valid]
            
            # 6. 結果を書き込み
            prediction_map[y_valid_start:y_valid_end, x_valid_start:x_valid_end] = valid_pred
            
            # 進捗表示
            # print(f"Processed tile: ({x}, {y})")

    return prediction_map

def predict_single_image(model, image):
    """
    1024x1024程度の画像に対して、一括で推論を行う関数
    """
    # 1. 特徴量の一括計算 (H, W, n_features)
    # 内部でパディング等は行われませんが、画像全体を使うので
    # 端っこのピクセル以外は正しく計算されます。
    features_map = _create_feature_stack(image)
    
    # 2. scikit-learn用に形状変換
    h, w, n_features = features_map.shape
    # (H*W, n_features) の2次元配列にする
    X = features_map.reshape(-1, n_features)
    
    # 3. NaN対策 (念のため)
    X = np.nan_to_num(X)
    
    # 4. 推論実行
    prediction_flat = model.predict(X)
    
    # 5. 画像形状に戻す
    prediction_map = prediction_flat.reshape(h, w)
    
    return prediction_map.astype(np.uint8)

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier

    # --- データの準備 (ダミー) ---


    # --- 1. 学習フェーズ ---
    print("Training phase...")
    from time import time
    start = time()
    
    # 特徴量抽出 (パッチサイズはフィルタ半径の2倍+1以上確保)
    num = 200
    for _ in range(30):
        H_img, W_img = 1000, 1000
        # RGB画像 (0-255)
        image = np.random.randint(0, 256, (H_img, W_img, 3), dtype=np.uint8)
        
        # 教師データ座標 (例: 100点)
        # 実際はGISツール等で作成したマスクから座標を取得する
        y_train_idx = np.random.randint(20, H_img-20, num)
        x_train_idx = np.random.randint(20, W_img-20, num)
        target_pixels = (y_train_idx, x_train_idx)
        
        # 正解ラベル (0:森, 1:水, 2:市街地 とする)
        y_labels = np.random.randint(0, 3, num)
        X_train = extract_training_features(image, target_pixels, patch_size=33)
    end = time()
    print(end - start )
    
    print(f"Feature shape: {X_train.shape}")
    
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_labels)
    print("Model trained.")

    # --- 2. 推論フェーズ ---
    print("Inference phase...")
    # 巨大画像を一括推論
    start = time()
    result_map = predict_single_image(clf, image)
    end = time()
    
    print(f"かかった時間:{end - start }")
    
    print(f"Result map shape: {result_map.shape}")
    print("Done.")
