# ニュース記事 RAG チャットボット

`yahoo_news_articles_preprocessed.csv`（このアプリ専用データ）を対象に、TF-IDFとSentenceTransformerによるハイブリッド検索＋Gemini APIで回答を生成するRAGチャットボット。

## ローカル実行

リポジトリ直下 (`streamlit/`) で以下を実行する。

```sh
pip install -r apps/rag_chatbot/requirements.txt
streamlit run apps/rag_chatbot/app.py
```

`GOOGLE_API_KEY` を環境変数（または `apps/rag_chatbot/.env`）に設定する必要がある。

## Streamlit Community Cloud にデプロイする場合

Main file path に `apps/rag_chatbot/app.py` を指定する。
Secrets に `GOOGLE_API_KEY` を設定する。
