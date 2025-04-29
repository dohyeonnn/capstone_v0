# full_pipeline_runner.py

from crawling import crawl_articles
from cleaner import clean_text
from summarize import summarize_text
from bertScore import bertscore
from verifier_roberta import verify


def run_pipeline(category="경제", limit=1):
    # print("\n Step 1: 뉴스 기사 크롤링")
    articles = crawl_articles(category, limit)

    for i, article in enumerate(articles):
        # print("\n" + "=" * 80)
        # print(f"제목 : [{i+1}] {article['title']}")
        # print(f"URL : {article['url']}")

        # Step 2: 전처리
        # print("\n Step 2: 본문 전처리")
        cleaned_text = clean_text(article["content_html"])
        print(f"원문 :\n{cleaned_text[:]}")

        # Step 3: 요약
        # print("\n Step 3: GPT 요약 실행")
        summary = summarize_text(cleaned_text)
        # print(f" 요약 결과:\n{summary}")

        # Step 4: BERTScore 측정
        # print("\n Step 4: BERTScore 계산")
        scores = bertscore(cleaned_text, summary)
        # print(
        #     f"BERTScore: F1={scores['f1']}, Precision={scores['precision']}, Recall={scores['recall']}"
        # )

        # Step 5: 2차 verification
        # print("\n Step 5: 요약본 2차 검증")
        verifyRet = verify(cleaned_text, summary)
        # print(f"검증 결과: {verifyRet}")

        print("=" * 80)


if __name__ == "__main__":
    run_pipeline(category="경제", limit=1)
