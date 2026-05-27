"""地震场景 Agent — 交互式 CLI"""

from agent import EarthquakeAgent


BANNER = """
╔══════════════════════════════════════════════╗
║          地震场景 AI 助手                     ║
║  Earthquake Scene AI Agent                   ║
╠══════════════════════════════════════════════╣
║  功能：                                       ║
║   · 地震数据查询与分析                        ║
║   · 中国地震台网实时速报                      ║
║   · 地震应急避险指导                          ║
║   · 地震科学知识问答                          ║
║   · 地震风险评估                              ║
╠══════════════════════════════════════════════╣
║  输入 'quit' 或 'exit' 退出                   ║
║  输入 'reset' 重置对话                        ║
║  输入 'help' 查看示例问题                     ║
╚══════════════════════════════════════════════╝
"""

HELP_TEXT = """
示例问题：

  [数据查询]
  · 最近中国有什么地震？
  · 查询过去一周四川5级以上地震
  · 查询2024年全球7级以上地震

  [应急指导]
  · 地震时在室内该怎么办？
  · 开车时遇到地震怎么处理？
  · 震后应该注意什么？

  [知识问答]
  · 什么是地震波？
  · 震级和烈度有什么区别？
  · 中国主要地震带有哪些？

  [风险评估]
  · 7级地震，我离震中100公里，有危险吗？
  · 8级地震对老旧建筑的影响
"""


def main():
    print(BANNER)

    try:
        agent = EarthquakeAgent()
    except ValueError as e:
        print(f"初始化失败: {e}")
        return

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("对话已重置。")
            continue

        if user_input.lower() == "help":
            print(HELP_TEXT)
            continue

        try:
            print("\n助手: ", end="", flush=True)
            reply = agent.chat(user_input)
            print(reply)
        except Exception as e:
            print(f"\n[错误] {e}")


if __name__ == "__main__":
    main()
