# verifier_roberta.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 1. GPU 사용 가능 여부 확인 및 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 모델 및 토크나이저 올리기
model = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForSequenceClassification.from_pretrained(model)
model.to(device)
model.eval()

# 3. 라벨 매핑
id2label = {0: "요약 품질 좋음", 1: "요약 품질 의심"}


def verify(premise: str, hypothesis: str) -> str:

    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(
        device
    )
    with torch.no_grad():
        output = model(**inputs)
        prob = F.softmax(output.logits, dim=-1)
        print("roberta softmax 값 : ", prob)
        predicted = torch.argmax(prob, dim=-1).item()
    return id2label[predicted]


s1 = "삼성바이오로직스의 1분기 영업 이익이 119.9% 상승하였다."
s2 = "삼성바이오로직스의 1분기 영업 이익이 119.9% 하락하였다"
print(f"문장 1: {s1}")
print(f"문장 2: {s2}")
result = verify(s1, s2)
print("roberta 검증 결과 : ", result)
