"""
Day 1 — LLM API Foundation
"""

import os
import time
from typing import Any, Callable

COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-6.1-terra": 0.005,
    "gpt-6-luna": 0.0005,
}

OPENAI_MODEL = "gpt-6.1-terra"
OPENAI_MINI_MODEL = "gpt-6-luna"


def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none",
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,
    )
    latency = time.perf_counter() - t0
    return resp.choices[0].message.content, latency


def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    return call_openai(prompt, model=OPENAI_MINI_MODEL, temperature=temperature, top_p=top_p, max_tokens=max_tokens)


def compare_models(prompt: str) -> dict:
    gpt4o_text, gpt4o_lat = call_openai(prompt)
    mini_text, mini_lat = call_openai_mini(prompt)
    est_tokens = len(gpt4o_text.split()) / 0.75
    cost = est_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS[OPENAI_MODEL]
    return {
        "gpt4o_response": gpt4o_text,
        "mini_response": mini_text,
        "gpt4o_latency": gpt4o_lat,
        "mini_latency": mini_lat,
        "gpt4o_cost_estimate": cost,
    }


def streaming_chatbot() -> None:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history = []
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in ("quit", "exit"):
            print("Bye!")
            break
        history.append({"role": "user", "content": user_input})
        stream = client.chat.completions.create(
            model=OPENAI_MINI_MODEL,
            messages=history,
            reasoning_effort="none",
            max_completion_tokens=256,
            stream=True,
        )
        print("Assistant: ", end="", flush=True)
        reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            reply += delta
        print()
        history.append({"role": "assistant", "content": reply})
        history = history[-6:]


def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(base_delay * 2 ** attempt)


def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results


def format_comparison_table(results: list[dict]) -> str:
    header = f"{'Prompt':<42} | {'GPT-4o Response':<42} | {'Mini Response':<42} | {'GPT-4o Latency':>14} | {'Mini Latency':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        p = r.get("prompt", "")[:40]
        g = r.get("gpt4o_response", "")[:40]
        m = r.get("mini_response", "")[:40]
        gl = r.get("gpt4o_latency", 0)
        ml = r.get("mini_latency", 0)
        lines.append(f"{p:<42} | {g:<42} | {m:<42} | {gl:>13.3f}s | {ml:>11.3f}s")
    return "\n".join(lines)


if __name__ == "__main__":
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    streaming_chatbot()
