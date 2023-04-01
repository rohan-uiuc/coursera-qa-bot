from utils import create_index, get_agent_chain, get_prompt_and_tools, get_search_index


def index():
    create_index()
    return True

def run(question):
    index = get_search_index()

    prompt, tools = get_prompt_and_tools()

    agent_chain = get_agent_chain(prompt, tools)

    return agent_chain.run(question)

