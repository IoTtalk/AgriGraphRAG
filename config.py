# The path of vector DB.
database_path = "chroma_0219"

# The path of enterprise data.
file_name = 'enterprises/chunks_20250219_cleaned.json'

# The chunk size used by text splitter.
chunk_size = 200

# Embedding model
# The embedding model repo from HuggingFace or model path from local.
embedding_model_path = 'dunzhang/stella_en_1.5B_v5'
# Use local finetuned model or not.
use_finetuned_model = False

# Directory name and file name of query file.
query_directory = "questions/"
query_file = "queries_1000.txt"

# Directory name of temp file while generating graph.
temp_directory = "temp/"

# File name of entity results file.
entity_file = "entity.json"

# Directory name of both retrieved results file and generated answers file.
output_directory = "results/0219/"

# File name of retrieved results file.
result_file = "graphRAG_Llama3.2-3B_1000Q_k10_Rtv.json"

# The number of retrieved results merged.
top_k = 10

# Configuration of Neo4j server
NEO4J_URI = "bolt://localhost:8687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Pcs54784"

# LLM model
# The LLM model repo from HuggingFace or model path from local.
llm_model_path = "meta-llama/Llama-3.2-3B-Instruct"

# File name of generated answers file.
answer_file = "graphRAG_Llama3.2-3B_1000Q_k10.json"
