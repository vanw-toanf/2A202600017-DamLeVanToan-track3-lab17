#!/usr/bin/env python3
"""
Multi-Memory Agent — Interactive CLI.

Chạy: python main.py
Gõ 'exit' hoặc 'quit' để thoát.
Gõ 'status' để xem trạng thái các memory backends.
Gõ 'reset' để xóa toàn bộ memory.
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from agent.graph import run_agent, get_memories, reset_memories


def print_memory_status():
    """In trạng thái của các memory backends."""
    short_term, long_term, episodic, semantic = get_memories()
    print("\n─── Memory Status ───────────────────────────────────────")
    print(f"  Short-term  : {short_term}")
    print(f"  Long-term   : {long_term}")
    print(f"  Episodic    : {episodic}")
    print(f"  Semantic    : {semantic}")
    profile = long_term.get_profile()
    if profile:
        print("\n  📋 User Profile:")
        for k, v in profile.items():
            print(f"     {k}: {v}")
    episodes = episodic.get_recent_episodes(3)
    if episodes:
        print("\n  📖 Recent Episodes:")
        for ep in episodes:
            ts = ep["timestamp"][:10]
            print(f"     [{ts}] {ep['summary'][:60]}")
    print("─────────────────────────────────────────────────────────\n")


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║     Multi-Memory Agent với LangGraph (Lab #17)       ║")
    print("║     Model: Qwen2.5:7b via Cloudflare Tunnel          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print("Lệnh đặc biệt:")
    print("  'status' — xem trạng thái memory")
    print("  'reset'  — xóa toàn bộ memory")
    print("  'exit'   — thoát")
    print()

    llm_link = os.getenv("LLM_LINK", "http://localhost:11434/v1")
    print(f"LLM endpoint: {llm_link}")
    print()

    conversation_id = "interactive"
    turn = 0

    while True:
        try:
            user_input = input(f"[Turn {turn + 1}] Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "thoát"):
            print("Tạm biệt! Hẹn gặp lại.")
            break

        if user_input.lower() == "status":
            print_memory_status()
            continue

        if user_input.lower() == "reset":
            reset_memories()
            print("[Memory đã được reset]\n")
            turn = 0
            continue

        response = run_agent(user_input, conversation_id=conversation_id)
        print(f"\nAgent: {response}\n")
        turn += 1


if __name__ == "__main__":
    main()
