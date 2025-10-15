from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic_core.core_schema import model_field
from langchain.tools import  tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from todoist_api_python.api import TodoistAPI

load_dotenv()

todoist_api_key = os.getenv("TODOIST_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

todoist = TodoistAPI(todoist_api_key)
@tool
def add_tasks(task, desc):
    """Add a task to the tasks list"""
    todoist.add_task(content=task, description=desc)

@tool
def show_tasks():
    """Show all tasks from todoist. Use this tool when user wants to see their tasks. If user asks to show the tasks, e.g.
       'show me the tasks' print them in a bullet list format """
    result_paginator = todoist.get_tasks()
    tasks=[]
    for task_list in result_paginator:
        for task in task_list:
            tasks.append(task.content)
    return tasks

tools = [add_tasks, show_tasks]



llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = gemini_api_key,
    temperature = 0.3
)

system_prompt ="You are a helpful assistant.You will help the user to add the tasks, you will help the user to shoe the existing tasks"
print("Welcome to the Task Manager Agent! I will help you to add task please provide your task to add it in todoist app")

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

#chain = prompt | llm | StrOutputParser()
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent,tools=tools, verbose=True)

history =[]
while True:
    user_input = input("you :")
    # response = chain.invoke({"input": user_input})
    agent_executor_response = agent_executor.invoke({"input": user_input, "history":history})
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=agent_executor_response['output']))
