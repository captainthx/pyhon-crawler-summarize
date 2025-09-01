import os
import sys
from ollama import ChatResponse, AsyncClient
from tools import add_two_numbers, search, available_functions
import asyncio
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")


system_prompt = """
You are an AI assistant that can answer questions using either provided context or by calling available tools.  

Guidelines:
1. If the context is sufficient to answer the question, use it directly.  
2. If the context is missing, insufficient, or outdated, you MUST call the most relevant tool from the available tools.  
3. Always prefer calling a tool when the user's question requires external knowledge that is not present in the current context.  
4. Never say "I have no context" unless the tool call also fails to return relevant data.  
5. When calling a tool, choose the function and arguments carefully based on the user’s question.  
6. After getting the tool output, use it as context to generate a clear, concise answer.  

Format your response to the user:
- If you answer from context → provide the final answer clearly.  
- If you answer using a tool → explain the result naturally without mentioning the tool call.  
"""


async def call_llm(prompt: str):
    response: ChatResponse = await AsyncClient(host=OLLAMA_HOST).chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,
        tools=[add_two_numbers, search],
    )

    async for chunk in response:
        if chunk.message.content:
            print(chunk.message.content, end="", flush=True)
            yield chunk.message.content

        if chunk.message.tool_calls:
            for tool_call in chunk.message.tool_calls:
                if function_to_call := available_functions.get(tool_call.function.name):
                    args = tool_call.function.arguments
                    if asyncio.iscoroutinefunction(function_to_call):
                        output = await function_to_call(**args)
                    else:
                        output = function_to_call(**args)
                    # เอา output ของ tool ไปให้ LLM สรุปเป็นข้อความ
                    followup_prompt = (
                        f"This is the result of the tool.'{tool_call.function.name}': {output}\n"
                        "Please summarize the response to the user in a clear"
                    )

                    # เรียก LLM อีกครั้ง เพื่อสรุป
                    followup_response: ChatResponse = await AsyncClient(
                        host=OLLAMA_HOST
                    ).chat(
                        model=OLLAMA_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": followup_prompt},
                        ],
                        stream=True,
                    )

                    async for fchunk in followup_response:
                        if fchunk.message.content:
                            yield fchunk.message.content
                else:
                    yield f"\nNot found!\n"


def display():
    st.header("LLM Search⚡")
    prompt = st.text_input("search", placeholder="your question here ?")
    btn = st.button("search", icon="🔍")

    if prompt and btn:
        with st.spinner("Thinking...", show_time=True):
            response = call_llm(prompt)
            st.write_stream(response)


display()
