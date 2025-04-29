import requests
from bs4 import BeautifulSoup
import time
from cleaner import clean_text

## 실 사용시에는 함수 input 추가하고 데이터 조절할 수 있게

headers = {"User-Agent": "Mozilla/5.0"}

# 카테고리 페이지 딕셔너리리
section_url_list = {
    "정치": "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100",
    "경제": "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=101",
    "사회": "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=102",
    "생활/문화": "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=103",
    "세계": "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=104",
    "IT/과학": "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=105",
}


# category를 section_url_list의 키로 사용하여 URL을 가져오는 함수
def get_article_links(category):
    section_url = section_url_list.get(category)
    if not section_url:
        raise ValueError(f"[ERROR] 지원하지 않는 카테고리: {category}")
    res = requests.get(section_url, headers=headers)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    links = set()
    for a in soup.select("a[href^='https://n.news.naver.com/mnews/article']"):
        links.add(a["href"])
    return list(links)


# 기존 함수로는 링크만 가져오기 때문에 본문 내용을 가져오는 함수 추가
def get_real_article_url(preview_url):
    try:
        res = requests.get(preview_url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")

        # 기사 제목 추출
        title = ""
        title_h2 = soup.find("h2", id="title_area")
        if title_h2:
            span = title_h2.find("span")
            if span:
                title = span.get_text(strip=True)

        real_url_meta = soup.find("meta", property="og:url")
        if real_url_meta and "article" in real_url_meta["content"]:
            return real_url_meta["content"], title
        return None
    except:
        return None


def get_article_content(article_url):
    try:
        res = requests.get(article_url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup.find_all(attrs={"style": True}):
            tag.decompose()

        for tag in soup.find_all(class_=["img_desc", "img_alt", "media_end_summary"]):
            tag.decompose()

        candidate_ids = ["dic_area", "newsct_article", "articeBody", "article_body"]
        for cid in candidate_ids:
            content_div = soup.find("div", id=cid) or soup.find("div", class_=cid)
            if content_div and content_div.get_text(strip=True):
                return str(content_div)

        return "[본문을 찾을 수 없음]"
    except Exception as e:
        return f"[오류 발생] {e}"


def crawl_articles(category, limit):
    article_urls = get_article_links(category)
    # print(f"\n\n{category} 카테고리에서 수집된 기사 수: {len(article_urls)}")

    results = []
    for i, preview_url in enumerate(article_urls[:limit], 1):
        # print(f"{i}. 프리뷰 URL: {preview_url}")
        real_url, title = get_real_article_url(preview_url)
        if not real_url:
            print("본문 URL 추출 실패")
            continue

        content_html = get_article_content(real_url)
        results.append({"url": real_url, "title": title, "content_html": content_html})
        time.sleep(0.5)

    return results

    # 실행 예시


articles = crawl_articles("경제", 1)
# for a in articles:
#     text = clean_text(a["content_html"])
#     print(f"Token count: {len(tokens)}")
# print(f"\n원본 URL : {a['url']}")
# print(f"[제목] : {a['title']}")
# print(f"[원문] : {a['content_html']}\n")
# print(f"[분리 결과] : \n\n{clean_text(a['content_html'])}\n")

# 제목은 title, 본문은 content_html로 저장됨
# print(f"[원본 url] : {articles[0]['url']}")
# print(f"[제목] : {articles[0]['title']}")
# print(f"[원문] : {articles[0]['content_html']}\n")

# print(f"[tag 제거 버전] : {clean_text(articles[0]['content_html'])}\n")
