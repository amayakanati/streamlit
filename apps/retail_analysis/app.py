# Run this app with `streamlit run apps/retail_analysis/app.py`
# and visit http://localhost:8501 in your web browser.

from pathlib import Path

import pandas as pd
import plotly.express as px
import scipy.stats as stats
import streamlit as st

# apps/retail_analysis/app.py から見て、リポジトリ直下の dataset/ を参照する。
# カレントディレクトリに依存しないよう __file__ を起点に絶対パスを解決する。
DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"


def load_data():
    return pd.read_csv(DATASET_DIR / "dataset.csv")


def preprocess_data(dataset):
    # 1. 外れ値の消去
    dataset_mod = dataset.drop(dataset.index[[1, 7]]).reset_index(drop=True)
    dataset_mod = dataset_mod.drop(dataset_mod.index[[218, 280, 409]]).reset_index(drop=True)

    # 2. 欠損値の平均値補完
    dataset_mod["weight"] = dataset_mod["weight"].fillna(113.0)

    # 3. price_type の表記ゆれ修正
    def modify_price_type_value(x):
        if x == "通常価格":
            return "定価"
        elif x == "割引価格":
            return "割引"
        else:
            return x

    dataset_mod["price_type"] = dataset_mod["price_type"].apply(modify_price_type_value)

    # 4. occupancy を小数に変換
    def make_occupancy_str_float(x):
        return float(x.replace("%", ""))

    dataset_mod["occupancy"] = dataset_mod["occupancy"].apply(make_occupancy_str_float)

    # 5. price_type と item_type をダミー変数化
    dataset_preprocessed = pd.get_dummies(
        data=dataset_mod,
        columns=["price_type", "item_type"],
    )

    return dataset_preprocessed


def one_sample_t_test(data, popmean, column):
    alpha = 0.05

    t_stat, p_value = stats.ttest_1samp(
        data[column],
        popmean,
        alternative="greater",
    )

    if p_value < alpha:
        result = f"結果: p値が優位水準：{alpha}未満なので、帰無仮説を棄却する。平均は{popmean}より有意に大きいと言える。"
    else:
        result = f"結果: p値が優位水準：{alpha}以上なので、帰無仮説を棄却できない。平均が{popmean}より大きいとは結論付けられない。"

    return t_stat, p_value, result


def two_sample_t_test(df_a, df_b, column="purchase_price"):
    alpha = 0.05

    t_stat, p_value = stats.ttest_ind(
        df_a[column],
        df_b[column],
        equal_var=False,
    )

    if p_value < alpha:
        result = "判定: 有意差あり（帰無仮説を棄却する）"
    else:
        result = "判定: 有意差なし（帰無仮説を棄却できない）"

    return t_stat, p_value, result


def _purchase_price_from_a():
    return pd.DataFrame(
        [
            ["ID_DN23", 93], ["ID_DO11", 159], ["ID_DO23", 38], ["ID_DP11", 96],
            ["ID_DP23", 106], ["ID_DP59", 97], ["ID_DQ11", 82], ["ID_DQ23", 88],
            ["ID_DQ59", 49], ["ID_DR23", 45], ["ID_DR35", 104], ["ID_DR47", 93],
            ["ID_DR59", 78], ["ID_DS11", 47], ["ID_DS35", 38], ["ID_DS47", 71],
            ["ID_DS59", 59], ["ID_DT11", 79], ["ID_DT47", 46], ["ID_DT59", 126],
            ["ID_DU11", 90], ["ID_DU23", 75], ["ID_DU35", 39], ["ID_DU59", 77],
            ["ID_DV23", 100], ["ID_DW11", 57], ["ID_DX47", 77], ["ID_DX59", 51],
            ["ID_DY35", 15], ["ID_DY47", 74], ["ID_DZ35", 70], ["ID_DA01", 63],
        ],
        columns=["item_id", "purchase_price"],
    )


def _purchase_price_from_b():
    return pd.DataFrame(
        [
            ["ID_DH26", 83], ["ID_DH50", 79], ["ID_DI14", 87], ["ID_DI26", 68],
            ["ID_DI50", 71], ["ID_DJ26", 127], ["ID_DK38", 109], ["ID_DL26", 62],
            ["ID_DL38", 73], ["ID_DM14", 81], ["ID_DM50", 56], ["ID_DN02", 78],
            ["ID_DN50", 67], ["ID_DO25", 68], ["ID_DO38", 95], ["ID_DO50", 64],
            ["ID_DP13", 66], ["ID_DP25", 107], ["ID_DF21", 105], ["ID_DQ13", 56],
            ["ID_DG09", 109], ["ID_DC08", 119], ["ID_DR25", 102], ["ID_DS01", 82],
            ["ID_DS25", 95], ["ID_DT49", 93], ["ID_DU01", 66], ["ID_DD33", 106],
            ["ID_DR32", 107], ["ID_DV01", 68], ["ID_DF57", 107], ["ID_DV37", 65],
            ["ID_DV49", 82], ["ID_DW01", 105], ["ID_DM20", 106], ["ID_DW25", 83],
            ["ID_DW37", 109],
        ],
        columns=["item_id", "purchase_price"],
    )


def main():
    st.title("小売店舗ビジネスデータ分析")

    dataset = load_data()
    dataset_preprocessed = preprocess_data(dataset)

    st.subheader("purchase_priceのヒストグラム")
    fig = px.histogram(
        dataset_preprocessed,
        x="purchase_price",
        title="purchase_priceの分布",
    )
    st.plotly_chart(fig)

    st.subheader("1標本t検定")
    selected_mean = st.number_input("比較する平均値 (popmean)", value=70)

    if st.button("検定を実行"):
        t_stat, p_value, result = one_sample_t_test(
            data=dataset_preprocessed, popmean=selected_mean, column="purchase_price"
        )
        st.write(f"t値: {t_stat:.4f}, p値: {p_value:.4f}")
        st.write(result)

    st.subheader("2標本t検定")

    if st.button("A社 vs B社のpurchase_priceを比較"):
        purchase_price_from_a = _purchase_price_from_a()
        purchase_price_from_b = _purchase_price_from_b()

        t_stat, p_value, result = two_sample_t_test(
            purchase_price_from_a, purchase_price_from_b, column="purchase_price"
        )
        st.write(f"t値: {t_stat:.4f}, p値: {p_value:.4f}")
        st.write(result)


if __name__ == "__main__":
    main()
