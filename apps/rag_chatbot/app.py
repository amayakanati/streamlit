import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

# .envファイルをロードして環境変数を設定
load_dotenv()

# APIキーを環境変数から取得
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("APIキーが設定されていません。Google CloudのAPIキーを設定してください。")
    st.stop()

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=api_key)

# CSVファイルを読み込む関数
@st.cache_data
def load_data(csv_file_path):
    df = pd.read_csv(csv_file_path)
    return df

# TF-IDFモデルを構築する関数
@st.cache_resource
def build_tfidf_model(texts):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer

# SentenceTransformerの埋め込みモデルを取得する関数
@st.cache_resource
def get_embedding_model():
    # 軽量で日本語性能の高い多言語対応モデルなどを指定
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# テキストデータをベクトル化する関数
@st.cache_resource
def build_embedding_model(texts):
    model = get_embedding_model()
    # テキストリストをまとめてベクトル化 (numpy配列にする)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

# ハイブリッド検索を行う関数
def hybrid_search(query, tfidf_matrix, tfidf_vectorizer, embeddings, texts, df, top_n=3):
    """
    TF-IDF（キーワード）と SentenceTransformer（ベクトル）を組み合わせてハイブリッド検索を行う
    """
    model = get_embedding_model()
    
    # 1. TF-IDFによる類似度計算
    query_tfidf = tfidf_vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    # 2. 埋め込みベクトルによる類似度計算
    query_embedding = model.encode([query])
    embedding_scores = cosine_similarity(query_embedding, embeddings).flatten()
    
    # 3. スコアの正規化 (Min-Maxスケーリングなど簡易的な組み合わせ)
    def normalize(scores):
        min_s, max_s = np.min(scores), np.max(scores)
        if max_s - min_s == 0:
            return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)
        
    norm_tfidf = normalize(tfidf_scores)
    norm_embedding = normalize(embedding_scores)
    
    # 重み付けハイブリッドスコア (例: ベクトル検索 0.7, キーワード検索 0.3)
    hybrid_scores = 0.7 * norm_embedding + 0.3 * norm_tfidf
    
    # スコアが高い上位 top_n 件のインデックスを取得
    top_indices = np.argsort(hybrid_scores)[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        results.append({
            "index": idx,
            "title": df.iloc[idx].get("title", ""),
            "text": df.iloc[idx].get("text", ""),
            "score": float(hybrid_scores[idx])
        })
    return results

# チャット履歴を初期化する関数
def init_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

# チャット履歴を表示する関数
def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Geminiモデルを使って応答を生成する関数
def respond_with_gemini(query, results, top_n=3):
    client = get_genai_client()
    
    # 検索された記事（コンテキスト）を組み立てる
    context_text = ""
    for i, res in enumerate(results[:top_n]):
        context_text += f"--- 記事 {i+1} ---\nタイトル: {res['title']}\n本文: {res['text']}\n\n"
        
    prompt = f"""あなたはニュース記事に関する質問に答える親切なAIアシスタントです。
以下の参考情報を基にして、ユーザーからの質問にわかりやすく答えてください。
参考情報だけで答えられない場合は、その旨を正直に伝えてください。

【参考情報】
{context_text}

【ユーザーからの質問】
{query}
"""
    # 新しいSDKの呼び出し方 (client.models.generate_content)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text

# ==========================================
# Streamlitアプリのメイン
# ==========================================
st.title("📰 ニュース記事 RAG チャットボット")
st.write("Yahoo!ニュースのデータをもとに、ニュースに関する質問にお答えします。")

# 必要なデータをロードし、処理する（このアプリ専用データなので同ディレクトリ内を参照する）
csv_file_path = Path(__file__).resolve().parent / "yahoo_news_articles_preprocessed.csv"
df = load_data(csv_file_path)

# テキストデータを抽出（例として 'text_mod' または 'text' 列を使用。データ構造に合わせて調整してください）
# 画像では 'text_mod' や 'text' 列が存在するため、今回は 'text' 列を利用します
texts = df["text"].fillna("").tolist()

with st.spinner("モデルを準備中（TF-IDF & ベクトル化）です..."):
    tfidf_matrix, tfidf_vectorizer = build_tfidf_model(texts)
    embeddings = build_embedding_model(texts)

# チャット履歴の初期化・表示
init_chat_history()
display_chat_history()

# ユーザーからの入力を受け取る
user_input = st.chat_input("ニュース内容について質問してください（例：最近の経済動向は？）")
if user_input:
    # ユーザーの入力をチャット履歴に追加・表示
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # 検索を実行
    with st.spinner("関連するニュースを検索中..."):
        search_results = hybrid_search(user_input, tfidf_matrix, tfidf_vectorizer, embeddings, texts, df, top_n=3)
        
    # Geminiによる回答生成
    with st.spinner("回答を生成中..."):
        ai_response = respond_with_gemini(user_input, search_results, top_n=3)
        
        # 参照した記事の情報を少し添えると親切です
        ai_response += "\n\n**【参考にしたニュース記事】**\n"
        for res in search_results:
            ai_response += f"- {res['title']}\n"
            
    # アシスタントの応答をチャット履歴に追加・表示
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)