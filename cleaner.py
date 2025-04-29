from bs4 import BeautifulSoup
import re


def clean_text(html):
    # 1. HTML 태그 제거
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ")

    # 2. 괄호 ((), [], {}) 안의 내용 제거 - 하나의 패턴으로 처리
    brackets = [r"\([^)]*\)", r"\[[^]]*\]", r"\{[^}]*\}"]
    for pattern in brackets:
        text = re.sub(pattern, "", text)

    # 3. 특수문자 정리
    text = re.sub(r"[^가-힣a-zA-Z0-9\s.,?!·~%]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(
        r"\(?[가-힣\s]+=[가-힣\s]+?\)?\s+[가-힣]+\s+(기자|특파원)\s*=", "", text
    )
    text = re.sub(r"\(?연합뉴스\)?\s*[가-힣]+\s+(기자|특파원)\s*=", "", text)
    text = re.sub(r"\(?[가-힣]+\s+(기자|특파원)\)?\s*=", "", text)
    text = re.sub(r"[가-힣]+\s+(기자|특파원)(\s*입니다)?", "", text)

    return text
