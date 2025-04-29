import os
import requests
import zipfile
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 0. 모델 파일 없으면 구글 드라이브에서 zip 다운로드 후 압축 풀기
model_dir = "./roberta_kornli_final"
model_zip = "./roberta_kornli_final.zip"

if not os.path.exists(model_dir):
    url = "https://drive.google.com/uc?export=download&id=1R6c_Gcq595ii4ay32NPQMYt_Lp-bBEa8"
    response = requests.get(url)
    with open(model_zip, "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(model_zip, "r") as zip_ref:
        zip_ref.extractall("./")

    os.remove(model_zip)  # zip 파일 삭제

# 1. GPU 사용 가능 여부 확인 및 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 모델 및 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.to(device)
model.eval()

# 3. 라벨 매핑
id2label = {0: 1, 1: 0.5, 2: 0}


def verify(premise: str, hypothesis: str) -> str:
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(
        device
    )
    with torch.no_grad():
        output = model(**inputs)
        prob = F.softmax(output.logits, dim=-1)
        # print("roberta softmax 값 : ", prob)
        predicted = torch.argmax(prob, dim=-1).item()
    return id2label[predicted]


# 예시
# s1 = "카페인을 너무 많이 먹었나봐요 너무 심장이 빨리 뛰어서 수업을 듣기 힘들었어요. 그래서 시험을 망쳤어요."
# s2 = "푹 자서 몸이 뻐근해."
# print(f"문장 1: {s1}")
# print(f"문장 2: {s2}")
# result = verify(s1, s2)
# print("roberta 검증 결과 : ", result)
