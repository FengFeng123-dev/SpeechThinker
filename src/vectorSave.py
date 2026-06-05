import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from config.settings import settings

# from pdfminer.high_level import extract_pages
# from pdfminer.layout import LTTextContainer
# import re
pdf_path = Path(__file__).parent.parent / "data" / "2023级普通本科人才培养方案-理工医类.pdf"

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", " ","?","!",";","；"]
)
chunks = splitter.split_documents(docs)

# 过滤空内容
chunks = [chunk for chunk in chunks if chunk.page_content and isinstance(chunk.page_content, str)]

# 查看chunks类型有哪些
# for chunk in chunks:
#     print(type(chunk.page_content))
#     if len(chunk.page_content) <100:
#         print(chunk.page_content)

embeddings = OpenAIEmbeddings(
    base_url = settings.API_BASE,
    model = settings.EMBEDDING_MODEL,
    api_key = settings.API_KEY,
    # tiktoken_enabled = False,
    check_embedding_ctx_length = False
)
dbp = str(Path(__file__).parent.parent / "chroma_db")
vector_db = Chroma.from_documents(
    persist_directory=dbp,
    embedding=embeddings,
    documents=chunks
)
for i, chunk in enumerate(chunks[:5]):
    print(f"=== Chunk {i} ===")
    print(chunk.page_content)
    print()
