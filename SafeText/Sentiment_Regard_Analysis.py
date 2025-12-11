
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import re
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# from tqdm import tqdm
# import torch
#
# def load_data(filepath: str) -> pd.DataFrame:
#     """Load dataset from a CSV file."""
#     df = pd.read_csv(filepath)
#     print(f"Loaded dataset from {filepath} with {df.shape[0]} rows.")
#     return df
#
# # ================= Sentiment & Regard =================
#
# class SentimentClassifier:
#     def __init__(self):
#         device = 0 if torch.cuda.is_available() else -1
#         tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
#         model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
#         # Initialize sentiment analysis pipeline
#         self.sentiment_pipeline = pipeline(
#             "text-classification",
#             model=model,
#             tokenizer=tokenizer,
#             truncation=True,
#             device=device
#         )
#
#     def compute_sentiment(self, texts):
#         # Returns a list of sentiment labels for the given list of texts
#         results = self.sentiment_pipeline(texts, batch_size=8)
#         return [r["label"] for r in results]
#
# class RegardClassifier:
#     def __init__(self):
#         device = 0 if torch.cuda.is_available() else -1
#         tokenizer = AutoTokenizer.from_pretrained("sasha/regardv3")
#         model = AutoModelForSequenceClassification.from_pretrained("sasha/regardv3")
#         # Initialize regard analysis pipeline
#         self.regard_pipeline = pipeline(
#             "text-classification",
#             model=model,
#             tokenizer=tokenizer,
#             truncation=True,
#             device=device
#         )
#
#     def compute_regard(self, texts):
#         # Returns a list of regard labels for the given list of texts
#         results = self.regard_pipeline(texts, batch_size=8)
#         return [r["label"] for r in results]
#
# # ================= SDG16 危险二分类模型 =================
#
# class SDG16DangerClassifier:
#     def __init__(self, model_path: str = "./sdg16_bert_synthetic"):
#         device = 0 if torch.cuda.is_available() else -1
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForSequenceClassification.from_pretrained(model_path)
#         tokenizer.model_max_length = 512
#         # Initialize classification pipeline for danger detection
#         self.pipe = pipeline(
#             "text-classification",
#             model=model,
#             tokenizer=tokenizer,
#             device=device,
#             truncation=True,
#             max_length=512
#         )
#
#     def predict(self, texts, batch_size: int = 8):
#         """
#         返回每个文本被判定为“危险”(类1)的概率分数（0~1）。
#         """
#         results = self.pipe(texts, batch_size=batch_size)
#         scores = []
#         for r in results:
#             # r["score"] 是 pipeline 输出的预测类别的概率
#             # 若预测标签为危险类 (LABEL_1)，则直接取该概率；若为安全类 (LABEL_0)，则用 1 - score 得到危险类概率
#             label = str(r["label"])
#             prob = float(r["score"])
#             if label in ["LABEL_1", "1"]:
#                 prob_danger = prob
#             else:
#                 prob_danger = 1.0 - prob
#             # 将概率四舍五入保留4位小数（值域仍为0~1）
#             scores.append(round(prob_danger, 4))
#         return scores
#
# # ================= 主分析流程 =================
#
# def analyse_sentiment_and_regard(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
#     if text_col not in df.columns:
#         raise ValueError(f"Column '{text_col}' not found in DataFrame.")
#
#     # 1) Sentiment 分析
#     sentiment_classifier = SentimentClassifier()
#     sentiments = []
#     for text in tqdm(df[text_col], desc="Analyzing Sentiment"):
#         label = sentiment_classifier.compute_sentiment([str(text)])[0]
#         sentiments.append(label)
#     df["Sentiment"] = sentiments
#
#     # 2) Regard 分析
#     regard_classifier = RegardClassifier()
#     regards = []
#     for text in tqdm(df[text_col], desc="Analyzing Regard"):
#         label = regard_classifier.compute_regard([str(text)])[0]
#         regards.append(label)
#     df["Regard"] = regards
#
#     # 3) SDG16 危险概率计算（BERT 模型）
#     print("Running SDG16 danger classifier (BERT) on texts.")
#     sdg16_clf = SDG16DangerClassifier(model_path="./sdg16_bert_synthetic")
#     texts = df[text_col].astype(str).tolist()
#     danger_scores = sdg16_clf.predict(texts, batch_size=8)
#     df["SDG16_danger_score"] = danger_scores
#
#     # 注意：不再计算 SDG16_risk_level 和 SDG16_risk_label
#
#     return df
#
# # 脚本执行入口
# if __name__ == "__main__":
#     # 输入输出文件路径（可根据需要修改）
#     input_file = "test_no_urls_truncated_regard.csv"
#     output_file = "output_results.csv"
#
#     df_input = load_data(input_file)
#
#     # 可选：如果只需处理部分数据，可在此处截取
#     # 示例：只处理第2000到3999行
#     # df_input = df_input.iloc[2000:4000]
#
#     # 分析文本的 Sentiment, Regard 和 SDG16 危险得分
#     results = analyse_sentiment_and_regard(df=df_input, text_col="text")
#
#     # 最终只保留指定的输出列
#     results = results[["text", "text_with_marker", "Sentiment", "Regard", "SDG16_danger_score"]]
#     results.to_csv(output_file, index=False)
#     print(f"Saved results with text, text_with_marker, Sentiment, Regard, and SDG16_danger_score to {output_file}.")
#

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import torch

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset from {filepath} with {df.shape[0]} rows.")
    return df

# ================= Sentiment & Regard =================

class SentimentClassifier:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            device=device
        )

    def compute_sentiment(self, texts):
        results = self.sentiment_pipeline(texts, batch_size=8)
        return [r["label"] for r in results]

class RegardClassifier:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained("sasha/regardv3")
        model = AutoModelForSequenceClassification.from_pretrained("sasha/regardv3")
        self.regard_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            device=device
        )

    def compute_regard(self, texts):
        results = self.regard_pipeline(texts, batch_size=8)
        return [r["label"] for r in results]

# ================= SDG16 Danger Model =================

class SDG16DangerClassifier:
    def __init__(self, model_path: str = "./sdg16_bert_synthetic"):
        device = 0 if torch.cuda.is_available() else -1
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer.model_max_length = 512
        self.pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            truncation=True,
            max_length=512
        )

    def predict(self, texts, batch_size: int = 8):
        results = self.pipe(texts, batch_size=batch_size)
        scores = []
        for r in results:
            label = str(r["label"])
            prob = float(r["score"])
            if label in ["LABEL_1", "1"]:
                prob_danger = prob
            else:
                prob_danger = 1.0 - prob
            scores.append(round(prob_danger, 4))
        return scores

# ================= Main Analysis Flow =================

def analyse_sentiment_and_regard(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")

    sentiment_classifier = SentimentClassifier()
    sentiments = []
    for text in tqdm(df[text_col], desc="Analyzing Sentiment"):
        label = sentiment_classifier.compute_sentiment([str(text)])[0]
        sentiments.append(label)
    df["Sentiment"] = sentiments

    regard_classifier = RegardClassifier()
    regards = []
    for text in tqdm(df[text_col], desc="Analyzing Regard"):
        label = regard_classifier.compute_regard([str(text)])[0]
        regards.append(label)
    df["Regard"] = regards

    print("Running SDG16 danger classifier (BERT) on texts.")
    sdg16_clf = SDG16DangerClassifier(model_path="./sdg16_bert_synthetic")
    texts = df[text_col].astype(str).tolist()
    danger_scores = sdg16_clf.predict(texts, batch_size=8)
    df["SDG16_danger_score"] = danger_scores

    # Add tag column
    def classify_danger(score):
        if score > 0.1:
            return "potentially threatening_content"
        elif score > 0.01:
            return "potentially expressed_hostility"
        else:
            return ""

    df["danger_label"] = df["SDG16_danger_score"].apply(classify_danger)

    return df

# Script entry point
if __name__ == "__main__":
    # input_file = "test_no_urls_truncated_regard.csv"
    # output_file = "output_results.csv"

    input_file = "MGSD - Expanded.csv"
    output_file = "MGSD_results.csv"



    df_input = load_data(input_file)
    results = analyse_sentiment_and_regard(df=df_input, text_col="text")

    results = results[["text", "text_with_marker", "Sentiment", "Regard", "SDG16_danger_score", "danger_label"]]
    results.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}.")



