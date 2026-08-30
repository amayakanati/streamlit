# 小売店舗ビジネスデータ分析

`dataset/dataset.csv` を前処理し、purchase_price の分布確認・1標本/2標本t検定を行うアプリ。

## ローカル実行

リポジトリ直下 (`streamlit/`) で以下を実行する。

```sh
pip install -r apps/retail_analysis/requirements.txt
streamlit run apps/retail_analysis/app.py
```

## Streamlit Community Cloud にデプロイする場合

Main file path に `apps/retail_analysis/app.py` を指定する。
