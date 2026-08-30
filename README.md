# streamlit

streamlit(python) practice

複数のStreamlitアプリを `apps/` 配下でアプリごとに独立管理しています。
各アプリは自分の `requirements.txt` を持ち、単独で（別々のStreamlit Community Cloudデプロイとしても）実行できます。

```
streamlit/
  apps/
    elearning_analysis/   # Eラーニングデータ分析
      app.py
      requirements.txt
      README.md
    retail_analysis/      # 小売店舗ビジネスデータ分析
      app.py
      requirements.txt
      README.md
  dataset/                 # 全アプリ共通のデータセット
  README.md
```

## 新しいアプリを追加する場合

1. `apps/<app_name>/` ディレクトリを作成する。
2. `app.py`（エントリポイント）と `requirements.txt` を置く。
3. `dataset/` 内のCSVを使う場合は、`Path(__file__).resolve().parents[2] / "dataset"` のように
   `__file__` 起点の絶対パスで参照する（どのディレクトリから `streamlit run` してもデータを見つけられるようにするため）。
4. ローカル確認: `pip install -r apps/<app_name>/requirements.txt && streamlit run apps/<app_name>/app.py`
