import os
import pickle

from langchain import LLMChain, OpenAI
from langchain.agents import ConversationalAgent, AgentExecutor, Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader, UnstructuredHTMLLoader
import faiss
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings



pickle_file = "open_ai.pkl"
index_file = "open_ai.index"

gpt_3_5 = OpenAI(model_name='gpt-3.5-turbo',temperature=0)

embeddings = OpenAIEmbeddings()

chat_history = []

memory = ConversationBufferWindowMemory(memory_key="chat_history")

gpt_3_5_index = None

def get_search_index():
    global gpt_3_5_index
    if os.path.isfile(pickle_file) and os.path.isfile(index_file) and os.path.getsize(pickle_file) > 0:
        # Load index from pickle file
        with open(pickle_file, "rb") as f:
            search_index = pickle.load(f)
    else:
        search_index = create_index()

    gpt_3_5_index = search_index


def create_index():
    source_chunks = create_chunk_documents()
    search_index = search_index_from_docs(source_chunks)
    faiss.write_index(search_index.index, index_file)
    # Save index to pickle file
    with open(pickle_file, "wb") as f:
        pickle.dump(search_index, f)
    return search_index


def search_index_from_docs(source_chunks):
    # print("source chunks: " + str(len(source_chunks)))
    # print("embeddings: " + str(embeddings))
    search_index = FAISS.from_documents(source_chunks, embeddings)
    return search_index


def get_html_files():
    loader = DirectoryLoader('docs', glob="**/*.html", loader_cls=UnstructuredHTMLLoader, recursive=True)
    document_list = loader.load()
    return document_list


def fetch_data_for_embeddings():
    document_list = get_text_files()
    document_list.extend(get_html_files())
    print("document list" + str(len(document_list)))
    return document_list


def get_text_files():
    loader = DirectoryLoader('docs', glob="**/*.txt", loader_cls=TextLoader, recursive=True)
    document_list = loader.load()
    return document_list


def create_chunk_documents():
    sources = fetch_data_for_embeddings()

    splitter = CharacterTextSplitter(separator=" ", chunk_size=800, chunk_overlap=0)

    source_chunks = splitter.split_documents(sources)

    print("sources" + str(len(source_chunks)))

    return source_chunks


def get_qa_chain(gpt_3_5_index):
    global gpt_3_5
    return ConversationalRetrievalChain.from_llm(gpt_3_5, chain_type="stuff", get_chat_history=get_chat_history,
            retriever=gpt_3_5_index.as_retriever(), return_source_documents=True, verbose=True)

def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)


def generate_answer(question) -> str:
    global chat_history, gpt_3_5_index
    gpt_3_5_chain = get_qa_chain(gpt_3_5_index)
    result = gpt_3_5_chain(
        {"question": question, "chat_history": chat_history, "vectordbkwargs": {"search_distance": 0.4}})
    chat_history = [(question, result["answer"])]
    sources = []

    for document in result['source_documents']:
        source = document.metadata['source']
        sources.append(source.split('\\')[-1].split('.')[0])

    source = ',\n'.join(set(sources))
    return result['answer'] + '\nModules: ' + source


def get_agent_chain(prompt, tools):
    global gpt_3_5
    llm_chain = LLMChain(llm=gpt_3_5, prompt=prompt)
    agent = ConversationalAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory,
                                                     intermediate_steps=True)
    return agent_chain


def get_prompt_and_tools():
    tools = get_tools()

    prefix = """Have a conversation with a human, answering the following questions as best you can. Always try to use Vectorstore first. Your name is Coursera Bot because your knowledge base is Coursera course. You have access to the following tools:"""
    suffix = """Begin! If you used vectorstore tool, ALWAYS return a "SOURCES" part in your answer"
    
    {chat_history}
    Question: {input}
    {agent_scratchpad}
    sources:"""
    prompt = ConversationalAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    return prompt, tools


def get_tools():
    tools = [
        Tool(
            name="Vectorstore",
            func=generate_answer,
            description="useful for when you need to answer questions about the coursera course about 3D Printing.",
            return_direct=True
        )]
    return tools
