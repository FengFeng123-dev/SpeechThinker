from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from config.settings import settings
from pathlib import Path
embeddings = OpenAIEmbeddings(
    base_url=settings.API_BASE,
    model=settings.EMBEDDING_MODEL,
    api_key=settings.API_KEY,
    check_embedding_ctx_length=False
)

pdf_path = str(Path(__file__).parent.parent.parent / "chroma_db")
# ========== RAG 对联检索器 ==========
couplet_vectorstore = Chroma(
    persist_directory=pdf_path,
    embedding_function=embeddings
)

@tool
def search_knowledge(query: str) -> str:
    """在本地知识库中搜索相关《2023级普通本科人才培养方案-理工医类》信息。当用户询问可能包含在《2023级普通本科人才培养方案-理工医类》时调用此工具。

    Args:
        query: 搜索关键词或问题
    """
    # 1. 优先使用向量检索
    couplet_retriever = couplet_vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = couplet_retriever.invoke(query)

    if docs:
        results = [f"【检索结果{i}】{doc.page_content}" for i, doc in enumerate(docs, 1)]
        return "\n\n".join(results)

    # 2. 向量库无结果时，回退到基础字典
    knowledge_base = {
        "python": "Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年创建。",
        "langchain": "LangChain是一个用于开发由语言模型驱动的应用程序的框架，支持链式调用、工具集成和记忆管理。",
        "langgraph": "LangGraph是LangChain的扩展，用于构建有状态的、多角色的LLM应用，支持循环图和持久化。",
        "agent": "AI智能体（Agent）是能够自主感知环境、做出决策并执行动作以实现目标的系统，核心能力包括工具调用和记忆管理。",
        "rag": "RAG（检索增强生成）结合了信息检索和文本生成，先从知识库检索相关内容，再交给LLM生成回答，有效减少幻觉。",
        "llm": "大语言模型（LLM）是基于深度学习的文本生成模型，代表有GPT系列、Qwen、Claude等。",
        "向量数据库": "向量数据库专门存储和检索高维向量数据，常用于RAG场景中的语义搜索，主流产品有ChromaDB、Milvus、Pinecone等。",
    }
    fallback_results = [
        f"【{key}】{value}" for key, value in knowledge_base.items()
        if key in query.lower() or query.lower() in key
    ]
    if fallback_results:
        return "\n".join(fallback_results)

    return f"未找到与「{query}」相关的知识条目。"

if __name__ == "__main__":
    print(search_knowledge.invoke("计算机科学与技术"))
