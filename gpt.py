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
    model="gpt-4-1106-preview",
    temperature: float = 0,
    max_tokens: int = 32,
    **kwargs,
) -> str:
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    reply: str = response.choices[0].message.content
    return reply


def make_variant(blocks: List[str]):
    _ = "You are a machine learning expert coder. "
    _ += "You will be given several blocks of code. "
    _ += "Create a new block of code inspired by the given blocks. "
    _ += "The block of code should use the same model name. "
    for i, block in enumerate(blocks):
        _ += f"<block_{i}>{block}</block>"
    _ += "Reply only with valid code. Do not explain."
    return gpt_text(
        messages=[{"role": "user", "content": _}],
        temperature=0.7,
        max_tokens=256,
    )


def make_player_name() -> str:
    _ = "Pick a name for an player. "
    _ += "The name should be a single word and use only lowercase letters. "
    _ += "Be creative!"
    base_name = gpt_text(
        messages=[{"role": "user", "content": _}],
        temperature=1.5,
        max_tokens=3,
    )
    return f"{base_name}.{str(uuid.uuid4())[:6]}"