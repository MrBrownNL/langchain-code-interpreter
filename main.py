import os
from typing import Any

from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool

load_dotenv()


def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer."""

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    python_agent_tools = [PythonREPLTool()]
    llm = ChatOpenAI(temperature=0, model=os.environ.get("OPENAI_MODEL"))

    python_agent = create_react_agent(
        prompt=prompt,
        llm=llm,
        tools=python_agent_tools,
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=python_agent_tools, verbose=True)

    # TODO: Run code on local machine to use the whole file as context as it is too large for OpenAI
    csv_agent_executor: AgentExecutor = create_csv_agent(
        llm=llm,
        path="episode_info.csv",
        agent_type="openai-tools",
        verbose=True,
        allow_dangerous_code=True,
    )

    ### Router Grand Agent ###

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                            returning the results of the code execution, no need to import qrcodes or pillow packages,
                            DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor,
            description="""useful when you need to answer questions about episode_info.csv file.
                        takes and input the entire question and returns the answer after running pandas calculations"""
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=llm,
        tools=tools,
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    print(grand_agent_executor.invoke(
        {"input": "which seasons have 24 episodes?"}
    ))

    print(grand_agent_executor.invoke(
        {"input": "generate 3 qr-codes pointing to google.com and save them in de qrcodes folder"}
    ))


if __name__ == "__main__":
    main()
