import os
import shutil
import glob
import gc
from typing import List
import json

# Import langchain frameware to build knowledge graph for GraphRAG.
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from neo4j import GraphDatabase

# Import transformer to load finetuned embedding model from local.
import torch
from transformers import AutoModel, AutoTokenizer

# Import function to generate answer using llama3.
from generation import generate_with_loop

# Import configuration file
import config


neo4j_uri = config.NEO4J_URI
neo4j_user = config.NEO4J_USER
neo4j_passwd = config.NEO4J_PASSWORD


class MyEmbedding:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze(0).tolist())
        return embeddings

    def embed_query(self, query: str) -> List[float]:
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).tolist()
        return embedding


def create_graph(tx, chunk):
    tx.run(
        "MERGE (d:Document {id: $id, text: $text})",
        id = chunk.metadata["id"],
        text = chunk.page_content,
    )


def set_graph(file_name, chunk_size, use_finetuned, embedding_model, database_path):
    """
    Use vector store with embedding model to build vector DB.

    Parameters:
        file_name (str): The file name of the json file contains chunks.
        chunk_size (int): A number representing the approximate length of every chunk.
        use_finetuned (bool): A signal representing use finetuned embedding model or not.
        embedding_model (str): The repo name of the embedding model on HuggingFace or the directory path of the embedding model if the model is stored locally.
        database_path (str): The directory name of vector database.

    Returns:
        int: The number of chunks.
    """
    
    texts = []
    
    with open(file_name, "r") as input_file:
        data = json.load(input_file)
        for item in data:
            texts.append(item["content"])

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=40)

    chunks = text_splitter.create_documents(texts)
    print(len(chunks))
    print(chunks[0])
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["id"] = "doc_{}".format(i)

    if use_finetuned:
        embeddings_model = MyEmbedding(embedding_model)
    else:
        embeddings_model = HuggingFaceEmbeddings(
            model_name = embedding_model,
            model_kwargs = {'device': 'cuda'},
            encode_kwargs = {'normalize_embeddings': False}
        )
    
    # Store chunks in FAISS
    faiss_db = FAISS.from_documents(chunks, embeddings_model)
    faiss_db.save_local(database_path)
    
    with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_passwd)) as driver:
        with driver.session() as session:
            for chunk in chunks:
                session.execute_write(create_graph, chunk)
    
    return len(chunks)


def hybrid_retrieve(user_query, num, use_finetuned, embedding_model, database_path, k):
    """
    Retrieve the results from vector DB using smilarity search with score, and then compare the scores to select the best retrieved result.

    Parameters:
        user_query (str): The query given by user.
        num (int): The number of results get from similarity search.
        use_finetuned (bool): A signal representing use finetuned embedding model or not.
        embedding_model (str): The repo name of the embedding model on HuggingFace or the directory path of the embedding model if the model is stored locally.
        database_path (str): The directory name of vector database.
        k (int): The number of retrieved results merged.

    Returns:
        list[str]: The top k retrieved results which are ranked by score of similarity search.
    """
    
    if use_finetuned:
        embeddings_model = MyEmbedding(embedding_model)
    else:
        embeddings_model = HuggingFaceEmbeddings(
            model_name = embedding_model,
            model_kwargs = {'device': 'cuda'},
            encode_kwargs = {'normalize_embeddings': False}
        )
    
    # FAISS search
    faiss_db = FAISS.load_local(database_path, embeddings_model, allow_dangerous_deserialization=True)
    faiss_results = faiss_db.similarity_search(user_query, k=k)
    
    faiss_results = [result.page_content for result in faiss_results]
    
    # Graph search
    with GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_passwd)) as driver:
        with driver.session() as session:
            graph_results = session.run("""
                MATCH (d:Document)-[:CITES|SIMILAR_TO]->(related)
                WHERE d.text CONTAINS $query
                RETURN related.text, related.id
                LIMIT $k
            """, query=user_query, k=k).data()
    
    graph_results = [result["d.text"] for result in graph_results]

    return graph_results


# run this python file only when a new vector DB is going to be set up
if __name__ == "__main__":
    # =====Setting Here=====
    # Set the path of vector DB.
    database_path = config.database_path
    
    # =====Setting Here=====
    # These are parameters used to build vector DB.
    file_name = config.file_name
    chunk_size = config.chunk_size
    embedding_model = config.embedding_model_path
    use_finetuned = config.use_finetuned_model
    
    # Check if the database path exists and is not empty
    if not os.path.exists(database_path) or os.path.getsize(database_path) == 0:
        print("Database path does not exist or is empty. Running set_graph...")
        chunk_number = set_graph(file_name, chunk_size, use_finetuned, embedding_model, database_path)
        print("Number of chunks: {}".format(chunk_number))
    else:
        print("Database path exists and is not empty. No need to run set_graph.")
    
    # =====Setting Here=====
    # Directory name and file name of query file.
    query_dir = config.query_directory
    query_file = config.query_file

    with open(query_dir + query_file, 'r') as fr:
        user_queries = fr.read().split("\n")
    
    retrieved_results = []  # List to store all retrieved results.
    num = 50  # Number of similarity search results.

    # =====Setting Here=====
    # Directory name of both retrieved results file and generated answers file.
    output_dir = config.output_directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    # =====Setting Here=====
    # File name of retrieved results file.
    result_file = config.result_file
    
    json_results = []
    
    # =====Setting Here=====
    # The number of retrieved results merged.
    top_k = config.top_k
    
    # Retrieve document and get result for each query.
    for i, query in enumerate(user_queries):
        results = hybrid_retrieve(query, num, use_finetuned, embedding_model, database_path, top_k)
        
        retrieved_results.append(results)
        
        json_results.append({
            "qid": i,
            "query": query,
            "retrieved_context": results
        })
        
        gc.collect()
    
    with open(output_dir + result_file, 'w') as output_file:
        json.dump(json_results, output_file, indent=4)
        output_file.close()
    
    '''
    # Seperate retrieving and generating.
    # Read result file to fulfill "retrieved_results" list.
    # Also fulfill "json_results" list for writing answer to answer_file.
    with open(output_dir + result_file, "r") as retrieved_file:
        data = json.load(retrieved_file)
        for item in data:
            json_results.append(item)
            retrieved_results.append(item["retrieved_context"])
    '''
    
    # =====Setting Here=====
    # File name of generated answers file.
    answer_file = config.answer_file
    # LLM model path.
    llm_model_path = config.llm_model_path
    
    # Generate answer and write into answer file for each retrieved result.
    with open(output_dir + answer_file, "w") as output_file:
        for i in range(len(retrieved_results)):
            histories = []

            query = user_queries[i]
            retrieved_result = retrieved_results[i]
            
            prompt = f"Question: {query}\n\nRelated documents: "
            
            for j in range(top_k):
                prompt += ("\n" + f"{j}. " + retrieved_result)
            
            prompt += "\n\nGenerate a answer for me."

            # Call generating function to get generated answer.
            generation = generate_with_loop(llm_model_path, prompt, histories)

            answer = ""
            
            # Keep update answer until the whole answer has been generated.
            for ans in generation:
                answer = ans
            
            json_results[i]["answer"] = answer
        
        json.dump(json_results, output_file, indent=4)
        output_file.close()