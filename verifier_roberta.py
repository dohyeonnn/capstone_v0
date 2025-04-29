import os
import requests
import zipfile
import gdown
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_dir = "./roberta_kornli_final"
model_zip = "./roberta_kornli_final.zip"
gdrive_file_id = "1R6c_Gcq595ii4ay32NPQMYt_Lp-bBEa8"


def safe_download_and_extract_zip():
    try:
        if os.path.exists(model_dir):
            print("[INFO] 모델 폴더가 이미 존재합니다.")
            return

        # 구글 드라이브에서 gdown으로 zip 다운로드
        url = f"https://drive.google.com/uc?id={gdrive_file_id}"
        print("[INFO] 모델 zip 파일 다운로드 시작...")
        gdown.download(url, model_zip, quiet=False)

        print("[INFO] 압축 해제 중...")
        with zipfile.ZipFile(model_zip, "r") as zip_ref:
            zip_ref.extractall("./")

        print("[INFO] 압축 완료.")
        os.remove(model_zip)

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"압축 후 {model_dir} 폴더가 존재하지 않음.")

    except zipfile.BadZipFile:
        raise RuntimeError("압축 해제 중 오류 발생 (zip 파일 손상).")
    except Exception as e:
        raise RuntimeError(f"모델 다운로드 중 오류 발생: {e}")


# 1. 다운로드 및 압축 해제
safe_download_and_extract_zip()

# 2. 모델 로딩
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
