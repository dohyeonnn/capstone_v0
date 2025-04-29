import openai
import os
from crawling import crawl_articles
from cleaner import clean_text
from summarize import summarize_text
from bertScore import bertscore
from verifier_roberta import verify

# 1. 뉴스 기사 크롤링 및 전처리
articles = crawl_articles(category="경제", limit=1)
article = articles[0]
cleaned_text = clean_text(article["content_html"])
print(cleaned_text)

# ▶️ API 키 세팅
openai.api_key = os.getenv(
    "OPENAI_API_KEY",
    "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",
)  # 안전하게 관리 필수

# 2. AI 요약 실행
ai_summary = summarize_text(cleaned_text)

# 3. 사용자 요약 입력 받기
user_summary = input("기사를 100자 이내로 요약해주세요: ")

# 4. BERTScore 계산
scores = bertscore(ai_summary, user_summary)
pre = scores["precision"]
rec = scores["recall"]

# 5. 번호(.)를 기준으로 분리
ast = [s.strip() for s in ai_summary.split(".") if s.strip()]
ust = [s.strip() for s in user_summary.split(".") if s.strip()]

# (1) sum(Roberta(clean_text, ust)) / len(ust)
arr_1 = [verify(cleaned_text, u) for u in ust]
roberta_sum_1 = sum(arr_1)
part1 = roberta_sum_1 / len(ust) if len(ust) > 0 else 0

# (2) (sum(Roberta(user_summary, ast)) + rec) / (len(ast) + rec)
arr_2 = [verify(user_summary, a) for a in ast]
roberta_sum_2 = sum(arr_2)
part2 = (roberta_sum_2 + rec) / (len(ast) + rec) if (len(ast) + rec) > 0 else 0

# (3) (sum(Roberta(ai_summary, ust)) + pre) / (len(ust) + pre)
arr_3 = [verify(ai_summary, u) for u in ust]
roberta_sum_3 = sum(arr_3)
part3 = (roberta_sum_3 + pre) / (len(ust) + pre) if (len(ust) + pre) > 0 else 0

# 모두 합치면 체제 metric
global_metric = part1 + part2 + part3
print("=" * 100)
print(f"요약 점수: {(global_metric / 3 ) * 100} 점")
print("=" * 100)

# 7. 피드백 사용자를 위해 0.5 이하 배열 검색
problematic_sentences = []

for idx, score_value in enumerate(arr_2):
    if score_value <= 0.5:
        problematic_sentences.append(("누락/모호", ast[idx]))

for idx, score_value in enumerate(arr_3):
    if score_value <= 0.5:
        problematic_sentences.append(("불필요/오류", ust[idx]))

# 8. 피드백 API 호출
if problematic_sentences:
    feedback_prompt = f"AI의 요약본 : {ai_summary}, \n사용자의 요약본 : {user_summary}\n 이고, 다음 문장들을 검토해서 피드백을 작성해줘.\n"
    for tag, sentence in problematic_sentences:
        if tag == "누락/모호":
            feedback_prompt += (
                f"- (누락되거나 모호하게 요약된 내용일 가능성 있음) {sentence}\n"
            )
        elif tag == "불필요/오류":
            feedback_prompt += f"- (불필요한 내용 포함될 가능성 있음) {sentence}\n"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "너는 문서 피드백을 작성하는 전문 비서야.",
                },
                {"role": "user", "content": feedback_prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        feedback = response.choices[0].message["content"].strip()
    except Exception as e:
        feedback = f"[API 오류] {e}"
else:
    feedback = "피드백할 문장이 없습니다."

print(f"피드백 :\n{feedback}")
