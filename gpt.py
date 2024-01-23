# prompt to combine two players and create variants
# prompt to create variant from single player
# prompt to judge output of two players

import uuid
import os
from openai import OpenAI
from typing import List


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def gpt_text(
    messages,  # List[Dict[str, str]]
    model="gpt-4-turbo",
    temperature: float = 0,
    max_tokens: int = 32,
    **kwargs,
) -> str:
    response = client.chat_completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    reply: str = response.choices[0].message.content
    return reply


def make_variant(blocks: List[str]):
    _ = "You are a machine learning expert coder. You will be given several blocks of code. Create a new block of code inspired by the given blocks."
    for i, block in enumerate(blocks):
        _ += f"<block_{i}>{block}</block>"
    _ += "Reply only with valid code. Do not explain."
    return gpt_text(
        messages=[
            {"role": "user", "content": _},
        ],
        temperature=1.7,
        max_tokens=256,
    )


def make_organism_name() -> str:
    base_name = gpt_text(
        messages=[
            {
                "role": "user",
                "content": "Pick a name for an organism. The name should be a single word, be creative!",
            },
        ],
        temperature=3.0,
        max_tokens=8,
    )
    return f"{base_name}.{str(uuid.uuid4())[:6]}"


def judge(responses: List[str], criteria: str):
    pass
