import os

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete, openai_embed

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

WORKING_DIR = "./dickens"

rag = LightRAG(
    working_dir=WORKING_DIR,
    embedding_func=openai_embed,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

# Create query parameters
query_param = QueryParam(
    mode="hybrid",  # or other mode: "local", "global", "hybrid"
)

# Example 1: Using the default system prompt
response_default = rag.query(
    "What are the primary benefits of renewable energy?",
    param=query_param
)
print(response_default)

# Example 2: Using a custom prompt
custom_prompt = """
You are an expert assistant in environmental science. Provide detailed and structured answers with examples.
"""
response_custom = rag.query(
    "What are the primary benefits of renewable energy?",
    param=query_param,
    prompt=custom_prompt  # Pass the custom prompt
)
print(response_custom)