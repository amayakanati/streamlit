# Run this app with `streamlit run steamlit_web_app.py` and
# visit http://localhost:8501 in your web browser.

# 以下のライブラリを使用するため、事前にターミナル（コマンドプロンプト）でインストールしてください。

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import plotly.express as px

# Q: result_app_imageフォルダに格納されているPNG画像のようになるよう、以下のコードの...部分を埋めて、streamlitを用いたwebアプリケーションを完成させてください。

# 最後に作成する main 関数を実行することで、Streamlit を起動させます。
# main 関数内で使用する関数をあらかじめ作成しておくことで、コードを見やすくし、間違いにも気づきやすくすることができます。
# まずは、Q1～Q3 で 3 つの関数を作成していきましょう。

# Q1: 小売店舗ビジネスデータを前処理する関数 preprocess_data を、「1-3_Pythonによる統計モデリング」で行った内容を参考にして作成してください。
# ただし、最低限、以下の前処理機能を満たすように作成してください。
    # 外れ値の消去
    # 欠損値の平均値補完
    # price_type の表記ゆれ修正
    # price_type と item_type をダミー変数化
    # occupancy を小数に変換

def preprocess_data(dataset): # 必要に応じて引数を入れてください。

    # 1.外れ値の消去
    dataset_mod = dataset.drop(dataset.index[[1, 7]]).reset_index(drop=True)
    dataset_mod = dataset_mod.drop(dataset_mod.index[[218, 280, 409]]).reset_index(drop=True)

    # 2.欠損値の平均値補完
    dataset_mod['weight'] = dataset_mod['weight'].fillna(113.0)
    
    # 3. price_type の表記ゆれ修正
    def modify_price_type_value(x):
        if x == '通常価格':
            return '定価'
        elif x == '割引価格':
            return '割引'
        else:
            return x
    
    dataset_mod['price_type'] = dataset_mod['price_type'].apply(modify_price_type_value)
    
    # 5.文字列変換(occupancy を小数に)
    def make_ocuupancy_str_float(x):
        return float(x.replace('%', ''))

    dataset_mod['occupancy'] = dataset_mod['occupancy'].apply(make_ocuupancy_str_float)

    # 4.price_type と item_type をダミー変数化
    dataset_preprocessed = pd.get_dummies(
        data=dataset_mod, 
        columns=['price_type', 'item_type']
    )
    
    return dataset_preprocessed

# Q2: 1標本t検定を行う関数 one_sample_t_test を作成してください。
def one_sample_t_test(data, popmean, column): # 必要に応じて引数を入れてください。    

    ## 有意水準の指定
    alpha = 0.05

    ## t検定の実施
    t_stat, p_value = stats.ttest_1samp(
        data[column], 
        popmean,
        alternative='greater'
    )

    ## 結果の解釈
    if p_value < alpha:
        result = f"結果: p値が優位水準：{alpha}未満なので、帰無仮説を棄却する。平均は{popmean}より有意に大きいと言える。"
    else:
        result = f"結果: p値が優位水準：{alpha}以上なので、帰無仮説を棄却できない。平均が{popmean}より大きいとは結論付けられない。"

    return t_stat, p_value, result

# Q3: 2標本t検定を行う関数 two_sample_t_test を作成してください。
def two_sample_t_test(df_a, df_b, column='purchase_price'): # 必要に応じて引数を入れてください。
    
    ## 有意水準の指定
    alpha = 0.05

    ## t検定の実施
    t_stat, p_value = stats.ttest_ind(
        df_a[column],
        df_b[column],
        equal_var=False
    )

    ## 結果の解釈
    if p_value < alpha:
        result = "判定: 有意差あり（帰無仮説を棄却する）"
    else:
        result = "判定: 有意差なし（帰無仮説を棄却できない）"

    return t_stat, p_value, result
    

# Q4: Streamlit アプリを起動すると呼び出されるmain関数を作成してください。
# ただし、result_app_imageフォルダに格納されているPNG画像を参考に、以下の要件を満たすように作成してください。
    # 'Eラーニングデータ分析'と'小売店舗ビジネスデータ分析'の2つのタブを用意し、選択すると画面が遷移する機能
        # Eラーニングデータ分析
            # ① code_module ごとの受講生数を棒グラフで可視化する機能
            # ② 'CCC', '2014J', 'Exam' のスコアの分布をヒストグラムで表示する機能
            # ③ final_result ごとの合計スコアを箱ひげ図で可視化する機能
        # 小売店舗ビジネスタブ
            # ④ 作成した関数、preprocess_data、one_sample_t_test、two_sample_t_test全てを用いること。
            # ⑤ purchase_price のヒストグラムの表示
            # ⑥ 1標本t検定を実装し、ボタンを押すと結果が表示
            # ⑦ 2標本t検定を実装し、A社とB社の purchase_price を比較（「1-3_Pythonによる統計モデリング」の「2-2. 2標本のt検定」で行ったことを参考にしてください。）

    
def main():
    
    st.title("Eラーニングデータ分析 & 小売店舗ビジネスデータ分析")
    
    # タブの作成
    tab1, tab2 = st.tabs(["Eラーニングデータ分析", "小売店舗ビジネスデータ分析"])

    with tab1:
        student_info = pd.read_csv('dataset/student_info.csv')
        module_assessments = pd.read_csv('dataset/module_assessments.csv')
        student_assessment = pd.read_csv('dataset/student_assessment.csv')
        
        st.header("Eラーニングデータの分析")
        
        # ① code_module ごとの受講生数を棒グラフで可視化する機能
        st.subheader('code_module ごとの受講生数')
        
        
        count_df = student_info.groupby('code_module')["id_student"].count().reset_index()

        fig1 = px.bar(
            count_df,
            x='code_module', 
            y='id_student', 
            labels={'code_module': 'code_module', 'id_student': 'id_student'},
            title='code_moduleごとの受講整数'
        )
        st.plotly_chart(fig1)

        # ② 'CCC', '2014J', 'Exam' のスコアの分布をヒストグラムで表示する機能
        st.subheader('モジュールCCC,プレゼンテーション2014JのExamスコア')

        assessments = pd.merge(module_assessments,student_assessment,on='id_assessment',how='right')
        assessments_CCC_2014J_Exam = assessments.query(
            'code_module == "CCC" and code_presentation == "2014J" and assessment_type == "Exam"'
        ).reset_index(drop=True)
        
        fig2 = px.histogram(
            assessments_CCC_2014J_Exam,
            x='score',
            nbins=20,
            title='Examスコアの分布'
        )
        
        st.plotly_chart(fig2)
        
        # ③ final_result ごとの合計スコアを箱ひげ図で可視化する機能
        assessments['weighted_score'] = assessments['score'] * (assessments['weight']/100)
        
        student_total_score = (
            assessments
            .groupby(['id_student', 'code_module', 'code_presentation'])
            .agg({'weighted_score': 'sum'})
            .rename(columns={'weighted_score': 'total_score'}) # ここでリネーム
            .reset_index()
        )
        
        tmp = pd.merge(student_info, student_total_score, on=['id_student', 'code_module', 'code_presentation'], how='right')
        
        fig3 = px.box(
            tmp, 
            x='final_result', 
            y='total_score',
            category_orders={"final_result": ['Withdrawn', 'Fail', 'Pass', 'Distinction']},
            title='最終結果と合計スコアの関係'
        )
        st.plotly_chart(fig3)

    with tab2:
        # preprocess_data() を呼び出してデータを準備
        dataset = pd.read_csv("dataset/dataset.csv")
        dataset_preprocessed = preprocess_data(dataset)

        st.header('小売店舗ビジネスデータの分析')

        # ヒストグラムを表示
        # ⑤ purchase_price のヒストグラムの表示
        st.subheader('purchase_priceのヒストグラム')
        
        fig4 = px.histogram(
            dataset_preprocessed,
            x='purchase_price',
            title = 'purchase_priceの分布'
        )
        st.plotly_chart(fig4)

        # 1標本t検定の入力フォーム
        st.subheader('1標本t検定')
        
        ## 平均値の選択
        selected_mean = st.number_input(
            '比較する平均値 (popmean)', 
            value=70
        )
        
        ## 検定実行ボタン
        if st.button('検定を実行'):
            # 検定を実行し、結果を出力
            t_stat, p_value, result = one_sample_t_test(data = dataset_preprocessed, popmean = selected_mean, column = 'purchase_price')

            # 画面に結果を出力
            st.write(f"t値: {t_stat:.4f}, p値: {p_value:.4f}")
            st.write(result)

        # 2標本t検定
        st.subheader('2標本t検定')
        
        ## 検定実行ボタン
        if st.button('A社 vs B社のpurchase_priceを比較'):
            
            # データを定義
            purchese_price_from_a = pd.DataFrame([
                ['ID_DN23', 93],
                ['ID_DO11',159],
                ['ID_DO23',38],
                ['ID_DP11',96],
                ['ID_DP23',106],
                ['ID_DP59',97],
                ['ID_DQ11',82],
                ['ID_DQ23',88],
                ['ID_DQ59',49],
                ['ID_DR23',45],
                ['ID_DR35',104],
                ['ID_DR47',93],
                ['ID_DR59',78],
                ['ID_DS11',47],
                ['ID_DS35',38],
                ['ID_DS47',71],
                ['ID_DS59',59],
                ['ID_DT11',79],
                ['ID_DT47',46],
                ['ID_DT59',126],
                ['ID_DU11',90],
                ['ID_DU23',75],
                ['ID_DU35',39],
                ['ID_DU59',77],
                ['ID_DV23',100],
                ['ID_DW11',57],
                ['ID_DX47',77],
                ['ID_DX59',51],
                ['ID_DY35',15],
                ['ID_DY47',74],
                ['ID_DZ35',70],
                ['ID_DA01',63],
            ], columns=['item_id', 'purchase_price'])

            purchese_price_from_b = pd.DataFrame([
                ['ID_DH26', 83],
                ['ID_DH50', 79],
                ['ID_DI14', 87],
                ['ID_DI26', 68],
                ['ID_DI50', 71],
                ['ID_DJ26', 127],
                ['ID_DK38', 109],
                ['ID_DL26', 62],
                ['ID_DL38', 73],
                ['ID_DM14', 81],
                ['ID_DM50', 56],
                ['ID_DN02', 78],
                ['ID_DN50', 67],
                ['ID_DO25', 68],
                ['ID_DO38', 95],
                ['ID_DO50', 64],
                ['ID_DP13', 66],
                ['ID_DP25', 107],
                ['ID_DF21', 105],
                ['ID_DQ13', 56],
                ['ID_DG09', 109],
                ['ID_DC08', 119],
                ['ID_DR25', 102],
                ['ID_DS01', 82],
                ['ID_DS25', 95],
                ['ID_DT49', 93],
                ['ID_DU01', 66],
                ['ID_DD33', 106],
                ['ID_DR32', 107],
                ['ID_DV01', 68],
                ['ID_DF57', 107],
                ['ID_DV37', 65],
                ['ID_DV49', 82],
                ['ID_DW01', 105],
                ['ID_DM20', 106],
                ['ID_DW25', 83],
                ['ID_DW37', 109],
            ], columns=['item_id', 'purchase_price'])

            # 検定を実行し、結果を出力
            t_stat, p_value, result = two_sample_t_test(purchese_price_from_a, purchese_price_from_b, column='purchase_price')

            # # 画面に結果を出力
            st.write(f"t値: {t_stat:.4f}, p値: {p_value:.4f}")
            st.write(result)

if __name__ == '__main__':
    main()