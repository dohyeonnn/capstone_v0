import os
import requests
import zipfile
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 0. 경로 설정
model_dir = "./roberta_kornli_final"
model_zip = "./roberta_kornli_final.zip"
gdrive_url = "https://drive.google.com/uc?export=download&id=1R6c_Gcq595ii4ay32NPQMYt_Lp-bBEa8"  # 형님 드라이브 링크


def safe_download_and_extract_zip():
    try:
        # 폴더가 이미 있으면 다운로드 생략
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            print("[INFO] 모델 폴더가 이미 존재합니다. 다운로드 생략.")
            return

        print("[INFO] 모델 zip 파일 다운로드 시작...")
        with requests.get(gdrive_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(model_zip, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print("[INFO] 다운로드 완료. 압축 해제 중...")
        with zipfile.ZipFile(model_zip, "r") as zip_ref:
            zip_ref.extractall("./")

        print("[INFO] 압축 해제 완료.")
        os.remove(model_zip)

        # 검증: 모델 폴더가 생겼는지 확인
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"[ERROR] 압축 후 모델 폴더가 없습니다: {model_dir}"
            )

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 다운로드 실패: {e}")
        raise RuntimeError("모델 zip 파일 다운로드 중 문제가 발생했습니다.")
    except zipfile.BadZipFile:
        print("[ERROR] zip 파일이 손상되었습니다.")
        raise RuntimeError("압축 해제 중 오류 발생 (zip 파일 손상).")
    except Exception as e:
        print(f"[ERROR] 예기치 못한 에러 발생: {e}")
        raise RuntimeError("모델 로딩 초기화 중 예기치 못한 오류가 발생했습니다.")


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
