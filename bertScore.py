from bert_score import score
import torch


def bertscore(sentence1, sentence2, lang="ko", model_type="xlm-roberta-base"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    P, R, F1 = score(
        [sentence2],
        [sentence1],
        lang=lang,
        model_type=model_type,
        verbose=False,
        device=device,
    )

    return {
        "precision": round(P.item(), 4),
        "recall": round(R.item(), 4),
        "f1": round(F1.item(), 4),
    }


# 예시 사용
# s1 = "카페인을 너무 많이 먹었나봐요 너무 심장이 빨리 뛰어서 수업을 듣기 힘들었어요. 그래서 시험을 망쳤어요."
# s2 = "카페인을 너무 많이 먹었나봐요 너무 심장이 빨리 뛰어서 수업을 듣기 힘들었어요. 근데 시험을 잘봤어요."

# print(f"문장 1: {s1}")
# print(f"문장 2: {s2}")
# result = bertscore(s1, s2)
# print("bertscore 검증 결과 : ", result)
