import json


def load_aihub_json(file_path):
    with open(file_path, encoding="utf-8-sig") as f:
        return json.load(f)


def extract_event_sentences(aihub_json):
    return [
        event.get("sentence", "")
        for event in aihub_json.get("data", {}).get("event", [])
    ]


# 사용 예시
file_path = "ME_Sample_13_L_1119439.json"
d = load_aihub_json(file_path)
sentences = extract_event_sentences(d)
print(sentences)
