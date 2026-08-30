# Run this app with `streamlit run apps/elearning_analysis/app.py`
# and visit http://localhost:8501 in your web browser.

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# apps/elearning_analysis/app.py から見て、リポジトリ直下の dataset/ を参照する。
# カレントディレクトリに依存しないよう __file__ を起点に絶対パスを解決する。
DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"


def load_data():
    student_info = pd.read_csv(DATASET_DIR / "student_info.csv")
    module_assessments = pd.read_csv(DATASET_DIR / "module_assessments.csv")
    student_assessment = pd.read_csv(DATASET_DIR / "student_assessment.csv")
    return student_info, module_assessments, student_assessment


def plot_student_count_by_module(student_info):
    count_df = student_info.groupby("code_module")["id_student"].count().reset_index()

    fig = px.bar(
        count_df,
        x="code_module",
        y="id_student",
        labels={"code_module": "code_module", "id_student": "id_student"},
        title="code_moduleごとの受講生数",
    )
    return fig


def plot_exam_score_distribution(module_assessments, student_assessment):
    assessments = pd.merge(module_assessments, student_assessment, on="id_assessment", how="right")
    assessments_ccc_2014j_exam = assessments.query(
        'code_module == "CCC" and code_presentation == "2014J" and assessment_type == "Exam"'
    ).reset_index(drop=True)

    fig = px.histogram(
        assessments_ccc_2014j_exam,
        x="score",
        nbins=20,
        title="Examスコアの分布",
    )
    return fig


def plot_total_score_by_final_result(student_info, module_assessments, student_assessment):
    assessments = pd.merge(module_assessments, student_assessment, on="id_assessment", how="right")
    assessments["weighted_score"] = assessments["score"] * (assessments["weight"] / 100)

    student_total_score = (
        assessments.groupby(["id_student", "code_module", "code_presentation"])
        .agg({"weighted_score": "sum"})
        .rename(columns={"weighted_score": "total_score"})
        .reset_index()
    )

    merged = pd.merge(
        student_info,
        student_total_score,
        on=["id_student", "code_module", "code_presentation"],
        how="right",
    )

    fig = px.box(
        merged,
        x="final_result",
        y="total_score",
        category_orders={"final_result": ["Withdrawn", "Fail", "Pass", "Distinction"]},
        title="最終結果と合計スコアの関係",
    )
    return fig


def main():
    st.title("Eラーニングデータ分析")

    student_info, module_assessments, student_assessment = load_data()

    st.subheader("code_module ごとの受講生数")
    st.plotly_chart(plot_student_count_by_module(student_info))

    st.subheader("モジュールCCC, プレゼンテーション2014JのExamスコア")
    st.plotly_chart(plot_exam_score_distribution(module_assessments, student_assessment))

    st.subheader("最終結果ごとの合計スコア")
    st.plotly_chart(
        plot_total_score_by_final_result(student_info, module_assessments, student_assessment)
    )


if __name__ == "__main__":
    main()
