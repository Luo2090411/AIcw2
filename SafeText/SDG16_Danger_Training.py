import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 输入带标签的数据集 CSV 文件
INPUT_FILE = "SDG16_danger_dataset_rule_v2.csv"

df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=["text"])

# 将标签映射为二分类 0/1（如有必要）
def map_binary_label(raw):
    lab = str(raw).strip().lower()
    # 将表示 SDG17 危险的标签映射为 1，其余映射为 0
    if lab == "stereotype_sdg16" or lab == "1":
        return 1
    else:
        return 0

df["label"] = df["label"].apply(map_binary_label)
df_bin = df[["text", "label"]].copy()

# 划分训练集和测试集
train_df, test_df = train_test_split(
    df_bin,
    test_size=0.2,
    random_state=42,
    stratify=df_bin["label"]
)

# 转换为 Hugging Face 数据集格式
train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

# Tokenizer 和 Model 初始化
MODEL_NAME = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize_fn, batched=True)
test_ds = test_ds.map(tokenize_fn, batched=True)

train_ds = train_ds.rename_columns({"label": "labels"})
test_ds = test_ds.rename_columns({"label": "labels"})
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,  # 0 = safe, 1 = dangerous
    ignore_mismatched_sizes=True
)

# 定义评估指标计算函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_binary": f1_score(labels, preds, average="binary")
    }

# 训练参数设定
OUTPUT_DIR = "./sdg16_bert_synthetic"
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=20,
    load_best_model_at_end=True,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)












