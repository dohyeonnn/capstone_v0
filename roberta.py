# kor_nli_finetune.py

import os
import pandas as pd
from datasets import Dataset, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
)
import csv

# 1. 환경 설정
os.environ["WANDB_DISABLED"] = "true"

# 2. 데이터 경로
train_file_path = "./kornli_train.tsv"
dev_file_path = "./kornli_dev.tsv"

# 3. 데이터 로드
train_df = pd.read_csv(
    train_file_path, quoting=csv.QUOTE_NONE, sep="\t", engine="python"
)
dev_df = pd.read_csv(dev_file_path, quoting=csv.QUOTE_NONE, sep="\t", engine="python")

# 4. 라벨 매핑
label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
train_df["label"] = train_df["gold_label"].map(label_map)
dev_df["label"] = dev_df["gold_label"].map(label_map)

# 5. 유효한 데이터만 필터링
train_df = train_df[train_df["label"].notnull()][["sentence1", "sentence2", "label"]]
dev_df = dev_df[dev_df["label"].notnull()][["sentence1", "sentence2", "label"]]

# 6. Dataset 변환
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(dev_df)

# 7. 모델 및 토크나이저 로드
model_name = "klue/roberta-base"
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 3
config.problem_type = "single_label_classification"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)


# 8. 토크나이징 함수
def tokenize_function(examples):
    sentence1_list = [str(s) if s is not None else "" for s in examples["sentence1"]]
    sentence2_list = [str(s) if s is not None else "" for s in examples["sentence2"]]

    outputs = tokenizer(
        sentence1_list,
        sentence2_list,
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    outputs["labels"] = examples["label"]
    return outputs


# 9. Tokenizing
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# 10. Label int64 캐스팅
tokenized_train = tokenized_train.cast_column("label", Value("int64"))
tokenized_eval = tokenized_eval.cast_column("label", Value("int64"))

# 11. 포맷 세팅
tokenized_train.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
tokenized_eval.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

# 12. 학습 세팅
training_args = TrainingArguments(
    output_dir="./roberta_kornli_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=100,
    report_to="none",  # wandb 비활성화
)

# 13. Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
)

# 14. 학습
trainer.train()

# 15. 모델 저장
save_dir = "./roberta_kornli_final"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"✅ 학습 완료 및 저장 위치: {save_dir}")
