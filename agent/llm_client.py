"""
LLM Client wrapper cho Multi-Memory Agent.
Dùng OpenAI-compatible API gọi về Qwen2.5:7b trên Colab.
"""
import os
from typing import List, Dict, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("MODEL", "qwen2.5:7b")
DEFAULT_SYSTEM = "Bạn là một trợ lý AI hữu ích với bộ nhớ đa tầng. Hãy trả lời bằng tiếng Việt một cách chính xác và tự nhiên."


def _get_client() -> OpenAI:
    """Tạo OpenAI client với base_url từ .env."""
    llm_link = os.getenv("LLM_LINK", "http://localhost:11434/v1")
    return OpenAI(
        base_url=llm_link,
        api_key="ollama",
    )


def call_llm(
    messages: List[Dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> Tuple[str, int, int]:
    """
    Gọi LLM và trả về (response_text, prompt_tokens, completion_tokens).
    
    Args:
        messages: List các messages theo format OpenAI
        model: Model name (default: qwen2.5:7b)
        temperature: Sampling temperature
        max_tokens: Max tokens để generate
    
    Returns:
        (response_text, prompt_tokens, completion_tokens)
    """
    client = _get_client()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        content = response.choices[0].message.content or ""
        
        # Token tracking
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else estimate_tokens(str(messages))
        completion_tokens = usage.completion_tokens if usage else estimate_tokens(content)
        
        return content, prompt_tokens, completion_tokens
        
    except Exception as e:
        error_msg = f"[LLM Error] {e}"
        print(error_msg)
        return error_msg, 0, 0


def estimate_tokens(text: str) -> int:
    """
    Ước lượng số tokens từ text.
    Dùng tiktoken nếu có, fallback về word count * 1.3.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: ~1.3 tokens per word for Vietnamese
        return int(len(text.split()) * 1.3)


def extract_facts_from_text(text: str, model: str = DEFAULT_MODEL) -> Dict:
    """
    Dùng LLM để extract profile facts từ text.
    Trả về dict các key-value facts.
    LLM-based extraction với parse/error handling.
    """
    prompt = f"""Phân tích đoạn văn sau và trích xuất thông tin cá nhân của người dùng.
Trả về theo định dạng JSON với các key như: name, age, occupation, preferences, allergies, hobbies, location.
CHỈ trả về JSON, không thêm text khác.

Văn bản: {text}

JSON:"""

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Bạn là một AI chuyên trích xuất thông tin có cấu trúc. Chỉ trả về JSON hợp lệ."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=256,
        )
        content = response.choices[0].message.content or "{}"
        
        # Parse JSON với error handling
        import json
        import re
        
        # Tìm JSON block trong response
        json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        if json_match:
            facts = json.loads(json_match.group())
            # Filter bỏ các keys có value null hoặc rỗng
            return {k: v for k, v in facts.items() if v and v != "null"}
        return {}
        
    except Exception as e:
        print(f"[extract_facts] Error: {e}")
        return {}
