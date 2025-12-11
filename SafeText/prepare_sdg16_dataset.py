import pandas as pd
import re

INPUT_FILE = "train_regard.csv"

# df = pd.read_csv(INPUT_FILE, nrows=2000)
df = pd.read_csv(INPUT_FILE)
print("Loaded:", INPUT_FILE, "rows:", len(df))


EXTREME_HATE = [
    "parasite", "vermin", "cancer", "disease", "cockroach",
    "nigger", "faggot", "pig", "rats"
]

NEGATIVE_STEREOTYPES = [
    "lazy", "greedy", "evil", "dirty", "corrupt"
]

# 一般辱骂词：也归入 abusive
INSULT = [
    "stupid", "idiot", "dumb", "moron", "worthless",
    "useless", "trash", "disgusting", "filthy", "hate",
    "hateful", "loser"
]

def contains_any(text: str, words) -> bool:
    """
    用正则进行词匹配，避免子串误判，例如 'rat' 不应匹配 'rational'。
    """
    if not isinstance(text, str):
        return False
    t = text.lower()
    if not words:
        return False
    pattern = r"\b(" + "|".join(re.escape(w) for w in words) + r")\b"
    return re.search(pattern, t) is not None

def sentiment_score(sentiment: str) -> int:
    """
    将 Sentiment 标签映射为 -1 / 0 / 1
    negative -> -1, positive -> 1, 其他 -> 0
    """
    if not isinstance(sentiment, str):
        return 0
    s = sentiment.lower()
    if "neg" in s:
        return -1
    if "pos" in s:
        return 1
    return 0

def regard_score(regard: str) -> int:
    """
    将 Regard 标签映射为 -1 / 0 / 1
    negative -> -1, positive -> 1, 其他 -> 0
    """
    if not isinstance(regard, str):
        return 0
    r = regard.lower()
    if "neg" in r:
        return -1
    if "pos" in r:
        return 1
    return 0

def compute_sdg16_risk_level(row) -> int:
    """
    输出三等级风险：
      2 -> very_dangerous （煽动仇恨 / 严重歧视言论）
      1 -> abusive        （一般冒犯性刻板印象 / 仇视言论）
      0 -> neutral        （其他）
    """
    text = str(row.get("text", "")).lower()
    cat = str(row.get("category", "")).lower()
    sent = str(row.get("Sentiment", "")).lower()
    regard = str(row.get("Regard", "")).lower()

    # 计算情感和态度的分值
    s_score = sentiment_score(sent)
    r_score = regard_score(regard)
    is_negative = (s_score < 0) or (r_score < 0)

    # ====== Level 2: very_dangerous ======
    # 含有极端仇恨词汇，且情感/态度负面 或 本身就是 stereotype
    if contains_any(text, EXTREME_HATE) and (is_negative or cat == "stereotype"):
        return 2

    # ====== Level 1: abusive ======
    # 含有冒犯性刻板印象词 或 一般辱骂词 -> 至少算 1
    if contains_any(text, NEGATIVE_STEREOTYPES) or contains_any(text, INSULT):
        # 如果还伴随 stereotype 或 负面情绪，可视为更严重的冒犯
        return 1

    # 其他情况默认判为 0
    return 0

def map_risk_label(level: int) -> str:
    if level == 2:
        return "very_dangerous"
    elif level == 1:
        return "abusive"
    return "neutral"

# 计算三等级风险
df["sdg16_risk_level"] = df.apply(compute_sdg16_risk_level, axis=1)
df["sdg16_risk_label"] = df["sdg16_risk_level"].apply(map_risk_label)

# 给出二分类危险标签：level > 0 则标记为 1，否则 0
df["label"] = (df["sdg16_risk_level"] > 0).astype(int)

# 导出包含新标签的数据集
OUT_FILE = "SDG16_danger_dataset_rule_v2.csv"
cols = [
    "text",
    "label",                # 二分类危险(1) / 非危险(0)
    "sdg17_risk_level",     # 0 / 1 / 2
    "sdg17_risk_label",     # neutral / abusive / very_dangerous
    "stereotype_type",
    "category",
    "Sentiment",
    "Regard",
]
available_cols = [c for c in cols if c in df.columns]

dataset = df[available_cols].copy()
dataset.to_csv(OUT_FILE, index=False, encoding="utf-8")
print(f"Saved dataset with SDG16 risk labels to {OUT_FILE}")

