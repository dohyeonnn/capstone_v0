import streamlit as st
import openai
import os
from crawling import crawl_articles
from cleaner import clean_text
from summarize import summarize_text
from bertScore import bertscore
from verifier_roberta import verify

# 패키 설정
openai.api_key = os.getenv(
    "OPENAI_API_KEY",
    "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",
)

if st.button("크롤링 + 전처리 시작"):
    article = crawl_articles(category="경제", limit=1)[0]
    cleaned_text = clean_text(article["content_html"])
    ai_summary = summarize_text(cleaned_text)

    st.session_state.cleaned_text = cleaned_text
    st.session_state.ai_summary = ai_summary

if "ai_summary" in st.session_state:
    st.subheader("원문")
    st.text_area("Cleaned Text", st.session_state.cleaned_text, height=300)

    st.subheader("AI 요약")
    st.text_area("AI Summary", st.session_state.ai_summary, height=150)

    user_summary = st.text_area("사용자 요약 3문장 이내)")

    if st.button("결과 테스트") and user_summary:
        scores = bertscore(st.session_state.ai_summary, user_summary)
        pre = scores["precision"]
        rec = scores["recall"]

        ast = [s.strip() for s in st.session_state.ai_summary.split(".") if s.strip()]
        ust = [s.strip() for s in user_summary.split(".") if s.strip()]

        arr_1 = [verify(st.session_state.cleaned_text, u) for u in ust]
        part1 = sum(arr_1) / len(ust) if len(ust) > 0 else 0

        arr_2 = [verify(user_summary, a) for a in ast]
        part2 = (sum(arr_2) + rec) / (len(ast) + rec) if (len(ast) + rec) > 0 else 0

        arr_3 = [verify(st.session_state.ai_summary, u) for u in ust]
        part3 = (sum(arr_3) + pre) / (len(ust) + pre) if (len(ust) + pre) > 0 else 0

        global_metric = part1 + part2 + part3
        st.success(f"테스트 결과: {round((global_metric/3)*100, 2)} 점")

        # 피드백 조회
        problematic_sentences = []

        for idx, score_value in enumerate(arr_2):
            if score_value <= 0.5:
                if idx < len(ust):  # ✅ 추가
                    problematic_sentences.append(("누락/모호", ust[idx]))

        for idx, score_value in enumerate(arr_3):
            if score_value <= 0.5:
                if idx < len(ust):  # ✅ 추가
                    problematic_sentences.append(("불필요/오류", ust[idx]))

        if problematic_sentences:
            feedback_prompt = f"AI의 요약본 : {st.session_state.ai_summary}, \n사용자의 요약본 : {user_summary}\n ..."

            for tag, sentence in problematic_sentences:
                if tag == "누락/모호":
                    feedback_prompt += f"- (누락되거나 모호하게 요약된 내용일 가능성 있음) {sentence}\n"
                elif tag == "불필요/오류":
                    feedback_prompt += (
                        f"- (불필요한 내용 포함될 가능성 있음) {sentence}\n"
                    )

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "피드백을 작성하는 비서"},
                        {"role": "user", "content": feedback_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                )
                feedback = response.choices[0].message["content"].strip()
                st.subheader("피드백")
                st.text_area("Feedback", feedback, height=300)
            except Exception as e:
                st.error(f"[API 오류]: {e}")
        else:
            st.info("피드백할 문장 없음.")
