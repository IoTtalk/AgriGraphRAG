# The path of vector DB.
database_path = "faiss_index"

# The path of enterprise data.
file_name = 'enterprises/chunks_20250219_cleaned.json'

# The chunk size used by text splitter.
chunk_size = 200

# Embedding model
# The embedding model repo from HuggingFace or model path from local.
embedding_model_path = 'dunzhang/stella_en_1.5B_v5'

# Directory name and file name of query file.
query_directory = "questions/"
query_file = "queries_1000.txt"

# Directory name of both retrieved results file and generated answers file.
output_directory = "results/0219/"

# File name of retrieved results file.
result_file = "graphRAG_Llama3.2-3B_1000Q_k10_Rtv.json"

# The number of retrieved results merged.
top_k = 10

# Configuration of Neo4j server
NEO4J_URI = "bolt://localhost:8687"
NEO4J_USER = "AgriGraph"
NEO4J_PASSWORD = "Pcs54784"

# File name of generated answers file.
answer_file = "graphRAG_Llama3.2-3B_1000Q_k10.json"
