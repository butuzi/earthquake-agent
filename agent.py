"""地震 Agent 核心 — Claude API 调用与工具调度"""

import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv

from tools import (
    query_earthquakes,
    get_recent_earthquakes,
    analyze_earthquake_data,
    get_emergency_guide,
    assess_earthquake_risk,
    search_knowledge,
)

load_dotenv()

SYSTEM_PROMPT = """你是一位专业的地震场景 AI 助手，拥有地震数据查询、应急响应指导和地震知识问答能力。

你的职责：
1. 帮助用户查询全球和中国地震数据（实时速报、历史查询）
2. 提供地震发生时的应急避险指导
3. 解答地震相关的科学知识
4. 分析地震数据并提供趋势洞察

使用原则：
- 优先使用工具获取准确数据，不要编造地震数据
- 应急指导要简洁明确，给出可操作的步骤
- 地震知识要科学准确，避免传播地震谣言
- 对于地震预警/预报相关问题，要明确区分预警和预报的概念
- 不要发布任何地震预测信息，地震预报是世界性科学难题
- 数据可视化时，用表格和列表清晰展示

语言：使用中文与用户交流。"""

# 工具定义
TOOLS = [
    {
        "name": "query_earthquakes",
        "description": "通过 USGS FDSN API 查询全球地震数据，支持按时间、区域（省份名或经纬度）、震级范围筛选。可用于查询历史地震记录。",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "string",
                    "description": "查询起始时间，格式 YYYY-MM-DD，默认最近30天",
                },
                "end_time": {
                    "type": "string",
                    "description": "查询结束时间，格式 YYYY-MM-DD，默认今天",
                },
                "region": {
                    "type": "string",
                    "description": "区域名称，支持: 中国、四川、云南、新疆、西藏、台湾、甘肃、青海",
                },
                "min_magnitude": {
                    "type": "number",
                    "description": "最小震级，如 5.0",
                },
                "max_magnitude": {
                    "type": "number",
                    "description": "最大震级",
                },
                "limit": {
                    "type": "integer",
                    "description": "最大返回条数，默认20",
                },
                "order_by": {
                    "type": "string",
                    "enum": ["time", "magnitude"],
                    "description": "排序方式: time(时间) 或 magnitude(震级)",
                },
            },
        },
    },
    {
        "name": "get_recent_earthquakes",
        "description": "获取中国地震台网最新地震速报数据，覆盖国内地震（含中小地震）。优先从中国地震台网获取，不可用时回退到 USGS。",
        "input_schema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "返回条数，默认10",
                },
            },
        },
    },
    {
        "name": "analyze_earthquake_data",
        "description": "对地震数据进行统计分析，包括震级分布、深度统计、时间范围等。需要传入 query_earthquakes 返回的 JSON 数据。",
        "input_schema": {
            "type": "object",
            "properties": {
                "earthquakes_json": {
                    "type": "string",
                    "description": "query_earthquakes 返回的 JSON 字符串",
                },
            },
            "required": ["earthquakes_json"],
        },
    },
    {
        "name": "get_emergency_guide",
        "description": "获取地震应急避险指导，根据不同场景提供具体的避险步骤和要点。",
        "input_schema": {
            "type": "object",
            "properties": {
                "scenario": {
                    "type": "string",
                    "enum": [
                        "indoor", "outdoor", "driving", "mountain",
                        "building_collapse", "seaside", "post_earthquake",
                    ],
                    "description": "场景类型: indoor(室内), outdoor(室外), driving(驾车), mountain(山区), building_collapse(被困), seaside(海边), post_earthquake(震后)",
                },
            },
        },
    },
    {
        "name": "assess_earthquake_risk",
        "description": "评估地震风险等级，根据震级、距离和建筑类型给出风险评估和建议。",
        "input_schema": {
            "type": "object",
            "properties": {
                "magnitude": {
                    "type": "number",
                    "description": "地震震级",
                },
                "distance_km": {
                    "type": "number",
                    "description": "距震中的距离（公里），可选",
                },
                "building_type": {
                    "type": "string",
                    "enum": ["一般建筑", "老旧建筑", "抗震建筑", "高层建筑"],
                    "description": "建筑类型",
                },
            },
            "required": ["magnitude"],
        },
    },
    {
        "name": "search_knowledge",
        "description": "搜索地震知识库，返回与查询关键词相关的地震科普知识。可用于回答地震相关的概念性问题。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，如'地震波'、'震级'、'海啸'等",
                },
            },
            "required": ["query"],
        },
    },
]

# 工具函数映射
TOOL_FUNCTIONS = {
    "query_earthquakes": query_earthquakes,
    "get_recent_earthquakes": get_recent_earthquakes,
    "analyze_earthquake_data": analyze_earthquake_data,
    "get_emergency_guide": get_emergency_guide,
    "assess_earthquake_risk": assess_earthquake_risk,
    "search_knowledge": search_knowledge,
}


class EarthquakeAgent:
    """地震场景 Agent"""

    def __init__(self):
        api_key = os.getenv("MIMO_API_KEY")
        if not api_key:
            raise ValueError(
                "请设置 MIMO_API_KEY 环境变量\n"
                "可以创建 .env 文件，内容为: MIMO_API_KEY=your-key"
            )
        self.client = Anthropic(
            api_key=api_key,
            base_url="https://api.xiaomimimo.com/anthropic",
        )
        self.messages = []
        self.model = "mimo-v2.5-pro"

    def chat(self, user_message: str) -> str:
        """发送用户消息并获取回复（支持多轮工具调用）。"""
        self.messages.append({"role": "user", "content": user_message})

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.messages,
            )

            # 收集 assistant 的回复内容
            assistant_content = response.content
            self.messages.append({"role": "assistant", "content": assistant_content})

            # 如果不需要工具调用，直接返回文本回复
            if response.stop_reason == "end_turn":
                return self._extract_text(response.content)

            # 处理工具调用
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                # 将工具结果加入消息继续对话
                self.messages.append({"role": "user", "content": tool_results})

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """执行工具调用。"""
        func = TOOL_FUNCTIONS.get(tool_name)
        if not func:
            return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)

        try:
            result = func(**tool_input)
            return result
        except Exception as e:
            return json.dumps({"error": f"工具执行出错: {str(e)}"}, ensure_ascii=False)

    def _extract_text(self, content) -> str:
        """从 Claude 响应中提取纯文本。"""
        texts = []
        for block in content:
            if block.type == "text":
                texts.append(block.text)
        return "\n".join(texts)

    def reset(self):
        """重置对话历史。"""
        self.messages = []
