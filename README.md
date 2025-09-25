# k-foldfl
## このリポジトリの目的

連合学習を学習する際に便利なFlowerだが、k-foldの実装方法に関して説明が少ない。研究で必要であったため、Cifar10を使って土台の作成を行う

## 実行方法
```bash
# 仮想環境（任意）
python -m venv .venv
source .venv/bin/activate

# 依存関係のインストール
pip install -U pip
pip install -r requirements.txt

python run_sim.py
