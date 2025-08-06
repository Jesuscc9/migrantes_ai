# chat/rag.py
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
import redis, os

embeddings = GoogleGenerativeAIEmbeddings()
vectordb = Chroma(persist_directory="./vectorstore",
                  embedding_function=embeddings)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

system_prompt = (
    "Eres un asistente experto que ayuda a inmigrantes recién llegados a Monterrey, México. "
    "Respondes SIEMPRE en español, de forma clara y empática. "
    "Si no sabes la respuesta, dices «No dispongo de esa información» en lugar de inventarla. "
    "Utiliza la información relevante de contexto cuando sea útil."
)

BASE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),   # memoria
    ("human", "{question}")
])

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

def get_chain(conversation_id: str):
    """Devuelve una ConversationalRetrievalChain con memoria en Redis."""
    r = redis.from_url(os.getenv("REDIS_URL"))
    history = RedisChatMessageHistory(
        session_id=conversation_id,
        url=os.getenv("REDIS_URL")
    )
    memory = ConversationBufferMemory(chat_memory=history,
                                      return_messages=True,
                                      memory_key="chat_history")
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": BASE_TEMPLATE,
        }
    )
    return chain

