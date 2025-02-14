import os

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#WORKING_DIR = "./dickens"
WORKING_DIR = "./legisys"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    embedding_func=openai_embed,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

with open("./test.txt", encoding="utf-8") as f:
    rag.insert(f.read())

# # Perform naive search
# print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))

# # Perform local search
# print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))

# # Perform global search
# print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

# # Perform hybrid search
# print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid")))

# # Perform mix search (Knowledge Graph + Vector Retrieval)
# # Mix mode combines knowledge graph and vector search:
# # - Uses both structured (KG) and unstructured (vector) information
# # - Provides comprehensive answers by analyzing relationships and context
# # - Supports image content through HTML img tags
# # - Allows control over retrieval depth via top_k parameter
# print(rag.query("What are the top themes in this story?", param=QueryParam(
#     mode="mix")))