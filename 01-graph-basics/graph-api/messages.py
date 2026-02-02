"""
Messages are used to represent communication between agents, tools, and the system in a workflow.

HumanMessage -  User input
AIMessage -  AI model response
FunctionMessage -  Result from a function/tool
SystemMessage -  System -level instructions
ToolMessage -  Output from a tool (if available)
"""

from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage, SystemMessage

user_msg = HumanMessage(content="What's the weather today?")                    # HumanMessage: User asks a question
ai_msg = AIMessage(content="The weather is sunny.")                             # AIMessage: AI responds
func_msg = FunctionMessage(name="get_weather", result="sunny")                  # FunctionMessage: Function/tool returns a result
sys_msg = SystemMessage(content="You are a helpful assistant.")                 # SystemMessage: System gives instructions or prompts
tool_msg = FunctionMessage(name="weather_tool", result="Temperature is 75°F")   # ToolMessage: Tool output
