import json
from typing import List


def format_for_finetuning(prompts: List[str], completions: List[str], output_path: str):
    assert len(prompts) == len(completions), "input과 output 인덱스 길이 다름 확인 필요"

    with open(output_path, "w", encoding="utf-8") as f:
        for p, c in zip(prompts, completions):
            item = {
                "prompt": p.strip(),
                "completion": " " + c.strip(),  # OpenAI 요구사항: 앞에 공백 하나
            }
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"✅ 저장 완료: {output_path} ({len(prompts)}개 샘플)")


## test code
# premise = [
#     "BBC Future의 조사에서 돼지고기 지방이 세계에서 가장 건강한 음식 8위에 선정됐다."
# ]
# hypothesis = [
#     "영국 매체에 따르면, BBC 조사에서 돼지고기 지방이 세계에서 여덟 번째로 건강한 음식으로 뽑혔어요."
# ]


# format_for_finetuning(premise, hypothesis, "prompt_completion_exam.jsonl")
