"""
Benchmark: so sánh no-memory agent vs with-memory agent.
10 multi-turn conversation scenarios.
Output: BENCHMARK.md

Chạy: python benchmark/run_benchmark.py
"""
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm_client import call_llm, estimate_tokens
from agent.graph import run_agent, reset_memories
from memory.long_term import UserProfileMemory
from memory.episodic import EpisodicMemory


# ── No-memory agent ───────────────────────────────────────────────────────────

def run_no_memory_agent(user_input: str, history: List[Dict]) -> Tuple[str, int]:
    """Agent không có memory: chỉ dùng short-term conversation history trong request."""
    messages = history[-4:] + [{"role": "user", "content": user_input}]
    system = {"role": "system", "content": "Bạn là trợ lý AI. Trả lời ngắn gọn bằng tiếng Việt."}
    full_messages = [system] + messages
    response, pt, ct = call_llm(full_messages, max_tokens=300)
    return response, pt + ct


# ── Benchmark scenarios ───────────────────────────────────────────────────────

SCENARIOS = [
    {
        "id": 1,
        "name": "Profile recall — tên người dùng",
        "category": "profile_recall",
        "turns": [
            "Xin chào! Tên tôi là Toàn, tôi là kỹ sư phần mềm.",
            "Bạn đang làm gì thú vị vậy?",
            "Tôi đang học về AI agents.",
            "Bạn có thể giới thiệu về LangGraph không?",
            "Ừ, nghe hay đó!",
            "Bạn có nhớ tên tôi là gì không?",
        ],
        "eval_turn_index": 5,
        "expected_keyword": "Toàn",
    },
    {
        "id": 2,
        "name": "Conflict update — dị ứng thực phẩm",
        "category": "conflict_update",
        "turns": [
            "Cho bạn biết: Tôi bị dị ứng sữa bò.",
            "Bạn có thể gợi ý thực phẩm thay thế sữa bò không?",
            "À nhầm, thực ra tôi dị ứng đậu nành chứ không phải sữa bò.",
            "Vậy tôi dị ứng gì vậy, bạn có nhớ không?",
        ],
        "eval_turn_index": 3,
        "expected_keyword": "đậu nành",
    },
    {
        "id": 3,
        "name": "Episodic recall — sự kiện debug Docker",
        "category": "episodic_recall",
        "turns": [
            "Tôi vừa fix xong bug Docker: dùng service name thay localhost.",
            "Cảm ơn, vấn đề đã giải quyết xong rồi!",
            "Bây giờ mọi thứ đang chạy ổn.",
            "Bạn có nhớ lần trước tôi fix bug gì không?",
        ],
        "eval_turn_index": 3,
        "expected_keyword": "docker",
    },
    {
        "id": 4,
        "name": "Semantic retrieval — FAQ về Docker",
        "category": "semantic_retrieval",
        "turns": [
            "Xin chào!",
            "Tôi đang gặp lỗi kết nối database trong Docker.",
            "Cách sử dụng Docker service name là như thế nào?",
        ],
        "eval_turn_index": 2,
        "expected_keyword": "service name",
    },
    {
        "id": 5,
        "name": "Trim / token budget — hội thoại dài",
        "category": "token_budget",
        "turns": [
            "Turn 1: Tôi tên là Minh.",
            "Turn 2: Tôi làm việc tại Hà Nội.",
            "Turn 3: Sở thích của tôi là đọc sách.",
            "Turn 4: Tôi thích lập trình Python.",
            "Turn 5: Gần đây tôi học về machine learning.",
            "Turn 6: Tôi cũng đang tìm hiểu về deep learning.",
            "Turn 7: Bạn có thể giải thích RAG là gì không?",
            "Turn 8: Thú vị thật!",
            "Turn 9: Bạn có nhớ tên tôi không?",
        ],
        "eval_turn_index": 8,
        "expected_keyword": "Minh",
    },
    {
        "id": 6,
        "name": "Profile — nghề nghiệp và sở thích",
        "category": "profile_recall",
        "turns": [
            "Tôi là bác sĩ, tôi thích chơi tennis.",
            "Công việc của tôi khá bận rộn.",
            "Cuối tuần tôi thường đi chơi thể thao.",
            "Bạn có biết nghề nghiệp của tôi là gì không?",
        ],
        "eval_turn_index": 3,
        "expected_keyword": "bác sĩ",
    },
    {
        "id": 7,
        "name": "Episodic — học LangGraph thành công",
        "category": "episodic_recall",
        "turns": [
            "Tôi vừa hoàn thành lab LangGraph xong rồi!",
            "Cảm giác rất tốt khi code chạy được.",
            "Hôm nay học được nhiều thứ.",
            "Bạn có nhớ tôi vừa hoàn thành gì không?",
        ],
        "eval_turn_index": 3,
        "expected_keyword": "langgraph",
    },
    {
        "id": 8,
        "name": "Semantic — giải thích RAG",
        "category": "semantic_retrieval",
        "turns": [
            "Tôi đang nghiên cứu về các kỹ thuật AI.",
            "RAG là gì, bạn có thể giải thích không?",
            "Tại sao RAG lại quan trọng?",
        ],
        "eval_turn_index": 1,
        "expected_keyword": "retrieval",
    },
    {
        "id": 9,
        "name": "Conflict — cập nhật địa chỉ",
        "category": "conflict_update",
        "turns": [
            "Tôi sống ở thành phố Hồ Chí Minh.",
            "Bạn biết tôi ở đâu không?",
            "À thực ra tôi vừa chuyển ra Hà Nội rồi.",
            "Bây giờ tôi đang ở đâu vậy?",
        ],
        "eval_turn_index": 3,
        "expected_keyword": "hà nội",
    },
    {
        "id": 10,
        "name": "Full memory — kết hợp nhiều loại",
        "category": "full_stack",
        "turns": [
            "Tên tôi là Lan, tôi làm data scientist.",
            "Tôi vừa deploy model lên production thành công.",
            "Hệ thống đang chạy ổn định.",
            "Bạn có thể giải thích ChromaDB là gì không?",
            "Cảm ơn thông tin hữu ích!",
            "Bạn có nhớ tên tôi và tôi vừa làm gì không?",
        ],
        "eval_turn_index": 5,
        "expected_keyword": "lan",
    },
]


# ── Run one scenario ──────────────────────────────────────────────────────────

def run_scenario_no_memory(scenario: Dict) -> Dict:
    """Chạy scenario với no-memory agent."""
    turns = scenario["turns"]
    eval_idx = scenario["eval_turn_index"]
    history = []
    result = {
        "responses": [],
        "total_tokens": 0,
        "eval_response": "",
        "pass": False,
    }

    for i, turn in enumerate(turns):
        resp, tokens = run_no_memory_agent(turn, history)
        history.append({"role": "user", "content": turn})
        history.append({"role": "assistant", "content": resp})
        result["total_tokens"] += tokens
        result["responses"].append(resp[:80])

        if i == eval_idx:
            result["eval_response"] = resp

    keyword = scenario.get("expected_keyword", "").lower()
    result["pass"] = keyword in result["eval_response"].lower() if keyword else False
    return result


def run_scenario_with_memory(scenario: Dict) -> Dict:
    """Chạy scenario với full-memory agent."""
    reset_memories()
    turns = scenario["turns"]
    eval_idx = scenario["eval_turn_index"]
    result = {
        "responses": [],
        "total_tokens": 0,
        "eval_response": "",
        "pass": False,
    }

    conv_id = f"bench_{scenario['id']}"
    for i, turn in enumerate(turns):
        resp = run_agent(turn, conversation_id=conv_id)
        result["total_tokens"] += estimate_tokens(turn) + estimate_tokens(resp)
        result["responses"].append(resp[:80])

        if i == eval_idx:
            result["eval_response"] = resp

    keyword = scenario.get("expected_keyword", "").lower()
    result["pass"] = keyword in result["eval_response"].lower() if keyword else False
    return result


# ── Generate BENCHMARK.md ─────────────────────────────────────────────────────

def generate_benchmark_md(rows: List[Dict]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# BENCHMARK.md — Multi-Memory Agent vs No-Memory Agent",
        "",
        f"> Benchmark chạy lúc: {now}",
        "> Model: Qwen2.5:7b (Colab Cloudflare tunnel)",
        "> Env: LangGraph + ChromaDB/Keyword fallback",
        "",
        "## Tổng quan",
        "",
        "| # | Scenario | Category | No-Memory Result | With-Memory Result | Pass (no-mem) | Pass (with-mem) |",
        "|---|----------|----------|------------------|--------------------|:---:|:---:|",
    ]

    pass_no = sum(1 for r in rows if r["no_memory"]["pass"])
    pass_with = sum(1 for r in rows if r["with_memory"]["pass"])

    for r in rows:
        sid = r["id"]
        name = r["name"]
        cat = r["category"]
        nm_resp = r["no_memory"]["eval_response"][:60].replace("\n", " ") or "(trống)"
        wm_resp = r["with_memory"]["eval_response"][:60].replace("\n", " ") or "(trống)"
        nm_pass = "✅" if r["no_memory"]["pass"] else "❌"
        wm_pass = "✅" if r["with_memory"]["pass"] else "❌"
        lines.append(f"| {sid} | {name} | {cat} | {nm_resp}... | {wm_resp}... | {nm_pass} | {wm_pass} |")

    lines += [
        "",
        f"**No-memory pass rate: {pass_no}/10**  ",
        f"**With-memory pass rate: {pass_with}/10**",
        "",
        "---",
        "",
        "## Chi tiết từng scenario",
        "",
    ]

    for r in rows:
        lines += [
            f"### Scenario {r['id']}: {r['name']}",
            f"- **Category:** {r['category']}",
            f"- **Expected keyword:** `{r['expected_keyword']}`",
            "",
            "**No-Memory response (eval turn):**",
            f"> {r['no_memory']['eval_response'][:200]}",
            "",
            "**With-Memory response (eval turn):**",
            f"> {r['with_memory']['eval_response'][:200]}",
            "",
            f"- No-memory: {'PASS ✅' if r['no_memory']['pass'] else 'FAIL ❌'}",
            f"- With-memory: {'PASS ✅' if r['with_memory']['pass'] else 'FAIL ❌'}",
            f"- Tokens used (approx): no-mem={r['no_memory']['total_tokens']}, with-mem={r['with_memory']['total_tokens']}",
            "",
            "---",
            "",
        ]

    lines += [
        "## Phân tích theo category",
        "",
        "| Category | Scenarios | With-Memory Pass |",
        "|----------|-----------|-----------------|",
    ]

    categories = {}
    for r in rows:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "pass": 0}
        categories[cat]["total"] += 1
        if r["with_memory"]["pass"]:
            categories[cat]["pass"] += 1

    for cat, stats in categories.items():
        lines.append(f"| {cat} | {stats['total']} | {stats['pass']}/{stats['total']} |")

    lines += [
        "",
        "---",
        "",
        "## Reflection — Privacy & Limitations",
        "",
        "### 1. Memory nào giúp agent nhất?",
        "**Short-term memory** đóng vai trò quan trọng nhất trong mỗi turn, giúp agent nhớ context hội thoại gần đây.",
        "**Long-term profile** giúp agent nhớ thông tin người dùng xuyên suốt session.",
        "",
        "### 2. Memory nào rủi ro nhất nếu retrieve sai?",
        "**Long-term profile** nguy hiểm nhất. Nếu extract sai facts (ví dụ: nhầm dị ứng sữa bò → lưu thành đậu nành trước khi user sửa),",
        "agent có thể đưa ra lời khuyên sức khỏe sai lệch. Đặc biệt nguy hiểm với thông tin y tế, tài chính.",
        "",
        "### 3. Nếu user yêu cầu xóa memory, xóa ở backend nào?",
        "- **Short-term**: Clear `ConversationBufferMemory` — dễ, chỉ cần reset deque.",
        "- **Long-term profile**: Xóa file `data/user_profile.json` hoặc gọi `UserProfileMemory.clear()`.",
        "- **Episodic**: Xóa file `data/episodic_log.json` hoặc filter theo user_id.",
        "- **Semantic**: Clear ChromaDB collection — cần xóa toàn bộ nếu không có user partitioning.",
        "",
        "> ⚠️ **PII Risk**: Long-term profile lưu tên, nghề nghiệp, dị ứng — thông tin nhạy cảm.",
        "> Cần có consent rõ ràng, TTL policy, và right-to-delete trước khi deploy production.",
        "",
        "### 4. Điều gì sẽ làm system fail khi scale?",
        "- **ChromaDB single-node**: không hỗ trợ multi-user isolation tốt, cần namespace per user.",
        "- **JSON file store** cho profile/episodic: không phù hợp concurrent access, cần migrate lên database.",
        "- **No TTL**: Episodic memory tích lũy không giới hạn → context window bị vượt nếu không trim.",
        "- **LLM extraction (extract_facts)**: Có thể extract sai không có human review, cần confidence threshold.",
        "- **Token budget tĩnh**: Budget 2000 tokens phù hợp Qwen2.5:7b nhưng cần điều chỉnh cho model khác.",
        "",
        "### 5. Limitations kỹ thuật hiện tại",
        "- Memory không có user_id partitioning → không thể multi-user.",
        "- Semantic embeddings dùng ChromaDB default model (all-minilm), không fine-tuned cho tiếng Việt.",
        "- Conflict resolution đơn giản (overwrite) — không có merge/diff cho structured data.",
        "- Không có async support → mỗi turn phải chờ LLM response xong.",
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Multi-Memory Agent Benchmark")
    print("=" * 60)
    print(f"Chạy {len(SCENARIOS)} scenarios...\n")

    rows = []
    for i, scenario in enumerate(SCENARIOS):
        print(f"\n[{i+1}/{len(SCENARIOS)}] Scenario {scenario['id']}: {scenario['name']}")
        print(f"  Category: {scenario['category']}")
        print(f"  Turns: {len(scenario['turns'])}")

        print("  → Chạy no-memory agent...")
        try:
            nm_result = run_scenario_no_memory(scenario)
        except Exception as e:
            print(f"  [ERROR] No-memory: {e}")
            nm_result = {"responses": [], "total_tokens": 0, "eval_response": f"ERROR: {e}", "pass": False}

        print("  → Chạy with-memory agent...")
        try:
            wm_result = run_scenario_with_memory(scenario)
        except Exception as e:
            print(f"  [ERROR] With-memory: {e}")
            wm_result = {"responses": [], "total_tokens": 0, "eval_response": f"ERROR: {e}", "pass": False}

        rows.append({
            "id": scenario["id"],
            "name": scenario["name"],
            "category": scenario["category"],
            "expected_keyword": scenario.get("expected_keyword", ""),
            "no_memory": nm_result,
            "with_memory": wm_result,
        })

        nm_status = "PASS ✅" if nm_result["pass"] else "FAIL ❌"
        wm_status = "PASS ✅" if wm_result["pass"] else "FAIL ❌"
        print(f"  No-memory: {nm_status} | With-memory: {wm_status}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    pass_no = sum(1 for r in rows if r["no_memory"]["pass"])
    pass_with = sum(1 for r in rows if r["with_memory"]["pass"])
    print(f"No-memory  pass rate: {pass_no}/10")
    print(f"With-memory pass rate: {pass_with}/10")

    # Write BENCHMARK.md
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "BENCHMARK.md")
    md_content = generate_benchmark_md(rows)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nBENCHMARK.md đã được ghi tại: {output_path}")

    return rows


if __name__ == "__main__":
    main()
