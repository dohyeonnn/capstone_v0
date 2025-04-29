import os
import openai
from typing import List

# ▶️ 실제 API 호출 여부
USE_FAKE_API = False  # ✅ 실사용 전환

# ▶️ API 키 세팅
openai.api_key = os.getenv(
    "OPENAI_API_KEY",
    "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",
)  # 안전하게 관리 필수


def summarize_text(text: str) -> str:
    """
    GPT-4o API를 사용해 긴 문서를 3문장 이내로 요약합니다.
    """
    if USE_FAKE_API:
        return f"[요약] {text[:30]}..."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "너는 문서를 요약하는 뉴스 요약 비서야.",
                },
                {
                    "role": "user",
                    "content": f"다음 문서를 핵심 정보를 빠뜨리지 않으면서 내용의 일관성, 간결성, 가독성을 고려해서 3문장 내외로 요약해줘.:\n{text}",
                },
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        return f"[API 오류] {e}"


def summarize_articles(texts: List[str]) -> List[str]:
    summaries = []
    for i, text in enumerate(texts):
        # print(f" 요약 중 ({i+1}/{len(texts)}):\n{text[:80]}...\n")
        summary = summarize_text(text)
        # print(f" 요약 결과:\n{summary}\n{'-'*80}\n")
        summaries.append(summary)
    return summaries
