# k-foldfl

## 目的
Flower は連合学習（Federated Learning; FL）を手早く構築できる一方、**k-fold による汎化性能の評価手順**は情報が少ない。  
本リポジトリは **CIFAR-10** を題材に、**FL × k-fold 検証の最小実装**（土台）を提供する。  
特に医療領域のようにデータが少量・分布ばらつきが大きい状況でも、**fold を跨いだ一貫した評価**でアルゴリズムを正しく比較できることを目的とする。

## 特長
- Flower（Server/Client）と **k-fold** を組み合わせた評価フロー
- 1 コマンドで **fold ごとの学習→評価** を実行可能

---

## アーキテクチャ概要
```
+-------------------+        Federated Rounds        +-------------------+
|   Flower Server   | <----------------------------> |   N Clients       |
| (strategy, eval)  |                                 | (per-fold data)   |
+-------------------+                                 +-------------------+
            ^                                                   ^
            | orchestrate k-fold                                | load fold_i
            +----------------- run_simulation.py ---------------+
```
- 既定では**全体データに K 分割**し、各 fold の学習・評価を順に実行  
---

## ディレクトリ構成
```
k-foldfl/
├─ run_simulation.py         # 実行エントリ（k-fold 制御）
├─ fl/
│  ├─ server.py              # 戦略/評価（例: FedAvg）
│  ├─ client.py              # 学習/評価ロジック
│  └─ task.py                # データダウンロード、モデル生成、Non-IID生成関数
├─ data/                     # （任意）ローカル配置先
├─ configs/
│  └─ default.yaml           # 既定設定（任意）
├─ requirements.txt
└─ README.md
```

---

## クイックスタート
```bash
# 仮想環境（任意）
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 依存関係
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt  # 無い場合は下の手動インストールを参照

# 1 fold を試す（既定設定）
python run_simulation.py
```

### 手動インストール（requirements.txt が無い場合）
```bash
python -m pip install -U pip wheel setuptools
pip install "flwr>=1.7,<2.0" torch torchvision numpy pandas scikit-learn
# GPU 版 PyTorch はご利用の CUDA に合わせて公式コマンドでインストールしてください
```

---

## 実行方法
```bash
python run_simulation.py

```

| 引数 | 既定値 | 説明 |
|---|---:|---|
| `--k` | 5 | fold 数（K） |
| `--fold` | 0 | 実行対象の fold（0..K-1）|
| `--num_clients` | 5 | クライアント数（擬似分割）|
| `--rounds` | 5 | 連合学習ラウンド数 |
| `--epochs` | 1 | 各ラウンドにおけるローカル学習エポック |
| `--batch_size` | 64 | ローカル学習のバッチサイズ |
| `--seed` | 42 | 乱数シード（分割・再現性に重要）|
| `--mode` | `global_kfold` | `global_kfold`：全体でK分割 / `client_kfold`：各クライアント内K分割 |
| `--out` | `runs/default` | ログ・成果物の保存先 |

---

## データセット（CIFAR-10）
- 既定では `torchvision.datasets.CIFAR10(download=True)` により自動取得

