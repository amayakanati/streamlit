# Eラーニングデータ分析

`dataset/student_info.csv`, `dataset/module_assessments.csv`, `dataset/student_assessment.csv` を使い、
受講生数・スコア分布・最終結果ごとの成績を可視化するアプリ。

## ローカル実行

リポジトリ直下 (`streamlit/`) で以下を実行する。

```sh
pip install -r apps/elearning_analysis/requirements.txt
streamlit run apps/elearning_analysis/app.py
```

## Streamlit Community Cloud にデプロイする場合

Main file path に `apps/elearning_analysis/app.py` を指定する。
