import logging
import os
from typing import Annotated, List, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from src.gemini_api import safe_invoke
from src.imbed import top_similarity

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    step: int


gemini_model = init_chat_model(
    model="gemini-3.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.7,
    model_provider="google_genai",
    timeout=60,
    max_retries=1,
)


def RAG_node(state: AgentState):
    retrieved_info = top_similarity(state["messages"][-1].content)
    logger.info(f"Retrieved:\n {[i[0] for i in retrieved_info]}")
    retrieved_content = "\n".join([i[0] for i in retrieved_info])
    return {"messages": [retrieved_content]}


def Answer_node(state: AgentState) -> AgentState:
    logger.info("Running Answer node...")
    system_prompt = """
    你是 楊孟翰 (Frank) 的推薦人，你想協助他獲得HR的面試邀請。
    若檢索到的資訊無法回答HR的問題，請直接說你不知道，無相關資訊可回答你的問題。
    若HR的問題與了 楊孟翰 (Frank) 的特質、背景、技能無關，或是涉及個人隱私，提醒他切勿詢問與求職者專業技能、背景不相關的資訊。
    任何與了解 楊孟翰 (Frank) 專業技能、人格特質、背景無關的問題，即使簡單無害也不要回覆，並提醒他只能詢問求職者專業技能、背景。
    楊孟翰是 Frank 的中文名，回覆時擇一即可。
    """
    query = f"""
    **HR的問題**
    {state['messages'][-2].content}

    **檢索到的資訊**
    {state['messages'][-1].content}

    請依據檢索到的資訊回答問題。

    **Example**
    回答HR的問題，用一句話描述 Frank 有什麼樣的人格特質、經歷或是專業技能，描述

    【哪項專業技能、特質】:在哪間公司的什麼經驗，可以給下一間公司的好處
    【哪項專業技能、特質】:在哪間公司的什麼經驗，可以給下一間公司的好處
    列出1至2項。
    """
    response = safe_invoke(
        gemini_model, [SystemMessage(system_prompt)] + [HumanMessage(query)]
    )
    # response = gemini_model.invoke(
    #     [SystemMessage(system_prompt)] + [HumanMessage(query)]
    # )
    logger.info("Answer node has been Completed!")
    return {
        "messages": [response],
        "step": state.get("step", 0) + 1,
    }


def Summary_node(state: AgentState) -> AgentState:
    logger.info("Running Summary node...")
    system_prompt = """
    你是 楊孟翰 (Frank) 的推薦人，幫他寫一篇 cover letter，協助他獲得HR的面試邀請。
    """
    previous_content = "\n".join([i.content for i in state["messages"]])
    query = f"""
    先前的問答
    {previous_content}

    依據先前的問答，幫 楊孟翰 (Frank) 寫一篇 cover letter。字數在300到400字。

    **Example**
    我覺得 楊孟翰 (Frank) 是一個適合的人選。
    說明為什麼要是 楊孟翰 (Frank) ？綜合先前HR的問題，著重在技能與人格特質可以如何幫助公司與團隊。
    """
    response = safe_invoke(
        gemini_model, [SystemMessage(system_prompt)] + [HumanMessage(query)]
    )
    # response = gemini_model.invoke(
    #     [SystemMessage(system_prompt)] + [HumanMessage(query)]
    # )
    logger.info("Summary node has been Completed!")
    return {"messages": [response]}


def summary_check(state: AgentState) -> Literal["Summary", "__end__"]:
    logger.info(f"Current Step: STEP {state['step']}.")
    if state["step"] >= 5:
        return "Summary"
    else:
        return "__end__"


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("RAG", RAG_node)
    graph.add_node("Answer", Answer_node)
    graph.add_node("Summary", Summary_node)

    graph.add_edge(START, "RAG")
    graph.add_edge("RAG", "Answer")
    graph.add_conditional_edges("Answer", summary_check)
    graph.add_edge("Summary", END)

    memory = MemorySaver()
    agent = graph.compile(checkpointer=memory)
    return agent
