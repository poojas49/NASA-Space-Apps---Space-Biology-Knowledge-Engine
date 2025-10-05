import json
import threading

import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid

TOKEN_LIMIT=4000
MAX_WORKERS = 4
from neo4j import GraphDatabase

GEMINI_API_KEY = ["AIzaSyBByU1sw8j2YoJbvVZ7hOATjyDTBRpVLSA"]  # Replace with your actual API key
g_l=len(GEMINI_API_KEY)
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key="
URI = "bolt://localhost:7687"  # Adjust if using a remote instance
USERNAME = "neo4j"
PASSWORD = "password"



class DocumentEmbeddingSystem:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333):
        # Qdrant connection
        self.client = QdrantClient(qdrant_host, port=qdrant_port)

        # Embedding model - using the specified model
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")

        # Get the actual vector size from the model
        self.vector_size = self.model.get_sentence_embedding_dimension()
        #print(f"Embedding dimension: {self.vector_size}")

    def create_collections(self):
        """Create a single collection for all documents"""
        collection_name = "document_chunks"

        # Check if collection exists
        collections = self.client.get_collections()
        collection_exists = collection_name in [c.name for c in collections.collections]

        if collection_exists:
            # If it exists, check if it has the correct vector size
            collection_info = self.client.get_collection(collection_name)
            existing_vector_size = collection_info.config.params.vectors.size

            if existing_vector_size != self.vector_size:
                # If sizes don't match, recreate the collection
                #print(f"Vector size mismatch: collection has {existing_vector_size}, model outputs {self.vector_size}")
                #print(f"Recreating collection {collection_name}...")
                self.client.delete_collection(collection_name)
                collection_exists = False

        if not collection_exists:
            # Create collection with the correct vector size
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            #print(f"Created collection '{collection_name}' with vector size {self.vector_size}")

        return collection_name

    def add_chunk(self, document_id, chunk_text, chunk_index=None, metadata=None):
        """
        Add a single chunk to a document

        Args:
            document_id: Identifier for the document
            chunk_text: Text content of the chunk
            chunk_index: Optional index/position of the chunk in the document
            metadata: Optional additional metadata as a dictionary

        Returns:
            The point ID stored in Qdrant
        """
        collection_name = self.create_collections()

        # If chunk_index not provided, try to determine next index
        if chunk_index is None:
            # Get current chunks for this document
            filter_param = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )

            existing_chunks = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_param,
                limit=10000
            )

            if existing_chunks[0]:
                # Find highest chunk_index and add 1
                max_index = max(point.payload.get("chunk_index", -1) for point in existing_chunks[0])
                chunk_index = max_index + 1
            else:
                # No existing chunks, start at 0
                chunk_index = 0

        # Generate embedding for the chunk
        embedding = self.model.encode(chunk_text)

        # Verify embedding dimension
        if len(embedding) != self.vector_size:
            raise ValueError(f"Expected embedding dimension {self.vector_size}, but got {len(embedding)}")

        # Convert to list for JSON serialization
        embedding = embedding.tolist()

        # Generate unique ID
        point_id = str(uuid.uuid4())

        # Prepare payload
        payload = {
            "document_id": document_id,
            "chunk_index": chunk_index,
            "text": chunk_text,
            "point_id": point_id
        }

        # Add custom metadata if provided
        if metadata and isinstance(metadata, dict):
            payload.update(metadata)

        # Create and store the point
        point = models.PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            return point_id
        except Exception as e:
            print(f"Error upserting point: {e}")
            print(f"Vector dimension: {len(embedding)}")
            raise

    def get_similar_chunks(self, query_text, document_id=None, limit=5):
        """
        Find chunks similar to the query text

        Args:
            query_text: The query to find similar chunks for
            document_id: Optional - filter results to a specific document
            limit: Maximum number of results to return

        Returns:
            List of similar chunks with metadata
        """
        collection_name = "document_chunks"

        # Generate embedding for query
        query_vector = self.model.encode(query_text).tolist()

        # Prepare filter if document_id is specified
        filter_param = None
        if document_id:
            filter_param = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )

        # Search in Qdrant
        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_param
        )

        # Format results
        results = []
        for res in search_results:
            # Extract all payload data
            result_data = {
                "point_id": res.id,
                "similarity_score": res.score
            }
            # Add all payload fields
            result_data.update(res.payload)
            results.append(result_data)

        return results

    def get_document_chunks(self, document_id):
        """
        Retrieve all chunks for a specific document

        Args:
            document_id: ID of the document to retrieve chunks for

        Returns:
            List of chunks in order of their chunk_index
        """
        collection_name = "document_chunks"

        # Create filter for document_id
        filter_param = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                )
            ]
        )

        # Query Qdrant
        scroll_results = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_param,
            limit=10000  # Adjust based on your expected document size
        )

        # Extract chunks with all metadata
        chunks = []
        for point in scroll_results[0]:
            chunk_data = {
                "point_id": point.id
            }
            chunk_data.update(point.payload)
            chunks.append(chunk_data)

        # Sort by chunk_index
        chunks.sort(key=lambda x: x["chunk_index"])

        return chunks

    def count_document_chunks(self, document_id):
        """Count the number of chunks in a document"""
        collection_name = "document_chunks"

        filter_param = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id)
                )
            ]
        )

        count = self.client.count(
            collection_name=collection_name,
            count_filter=filter_param
        )

        return count.count

counter_lock = threading.Lock()
COUNT=0
def increment_counter():
    global COUNT
    with counter_lock:
        current = COUNT
        COUNT += 1
        return current
# Example usage


def get_summaries_by_subcategory_id(subcategory_id):
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        query = """
        MATCH (s:SubAttribute {id: $subcategory_id})
        OPTIONAL MATCH (s)<-[:DESCRIBES]-(t:TextSummary)
        WITH s, collect(t) as summaries
        RETURN s.id as subcategory_id, 
               s.name as subcategory_name, 
               summaries
        """
        result = session.run(query, subcategory_id=subcategory_id)
        record = result.single()
        # print(record)
        if not record:
            return None

        word_count = 0
        summary_list = ""

        # Debug the record
        #print(f"SubAttribute: {record['subcategory_name']}")

        # The issue is in how you're accessing the summaries
        for summary in record["summaries"]:
            if summary is not None:
                # In Neo4j, the nodes are returned as Neo4j node objects
                # You need to access their properties using dictionary syntax
                if "content" in summary:
                    content = summary["content"]
                    summary_list += content + "\n"
                    word_count += len(content.split())

        #print("word count: " + str(word_count))
        return summary_list

# if __name__ == "__main__":
#     # # Initialize the system
#     system = DocumentEmbeddingSystem()
#     #
#     # # Example: Adding chunks incrementally to documents
#     #
#     # # Document 1 - Adding incrementally
#     # doc1_id = "incremental_doc_1"
#     #
#     # # Add chunks one by one
#     # print(f"Adding chunks to {doc1_id}...")
#     #
#     # chunk1_id = system.add_chunk(
#     #     document_id=doc1_id,
#     #     chunk_text="Artificial intelligence is transforming industries worldwide."
#     # )
#     # print(f"Added chunk 1 with ID: {chunk1_id}")
#     #
#     # chunk2_id = system.add_chunk(
#     #     document_id=doc1_id,
#     #     chunk_text="Machine learning models require large datasets to be effective."
#     # )
#     # print(f"Added chunk 2 with ID: {chunk2_id}")
#     #
#     # # Add a chunk with custom metadata
#     # chunk3_id = system.add_chunk(
#     #     document_id=doc1_id,
#     #     chunk_text="Neural networks are inspired by the human brain's structure.",
#     #     metadata={"importance": "high", "source": "textbook"}
#     # )
#     # print(f"Added chunk 3 with ID: {chunk3_id}")
#     #
#     # # Document 2 - Adding incrementally with specified indices
#     # doc2_id = "incremental_doc_2"
#     #
#     # # Add chunks with specific indices
#     # system.add_chunk(
#     #     document_id=doc2_id,
#     #     chunk_text="Python is a popular programming language for data science.",
#     #     chunk_index=0
#     # )
#     #
#     # system.add_chunk(
#     #     document_id=doc2_id,
#     #     chunk_text="Libraries like TensorFlow make machine learning easier in Python.",
#     #     chunk_index=2  # Intentionally skipping index 1
#     # )
#     #
#     # system.add_chunk(
#     #     document_id=doc2_id,
#     #     chunk_text="Data scientists often use Python libraries for analysis.",
#     #     chunk_index=1  # Adding the missing chunk later
#     # )
#     #
#     # # Count chunks in each document
#     # doc1_count = system.count_document_chunks(doc1_id)
#     # doc2_count = system.count_document_chunks(doc2_id)
#     #
#     # print(f"\nDocument 1 has {doc1_count} chunks")
#     # print(f"Document 2 has {doc2_count} chunks")
#     #
#     # # Query for similar chunks
#     # print("\nQuerying for AI-related content:")
#     # results = system.get_similar_chunks("How does artificial intelligence work?", limit=3)
#     # for i, res in enumerate(results):
#     #     print(f"{i + 1}. Document: {res['document_id']}, Score: {res['similarity_score']:.4f}")
#     #     print(f"   Text: {res['text']}")
#     #
#     # # Get all chunks from document 2 in order
#     # print("\nRetrieving all chunks from document 2:")
#     # chunks = system.get_document_chunks(doc2_id)
#     # for i, chunk in enumerate(chunks):
#     #     print(f"{i + 1}. Index {chunk['chunk_index']}: {chunk['text']}")

#     #ask for query and doubts

#     #query llm first.

#     #then query database.

#     #get data from neo4j.

#     #get data from profile.

#     #send everything to llm.

#     #parse and refnie the data from llm.

#     user_input = input("Please enter your question: ")

#     your_question = (
#         f'The user has asked a question: "{user_input}"\n\n'
#         "Please improve and refine the question to make it:\n"
#         "- Clearer\n"
#         "- More detailed if needed\n"
#         "- Grammatically correct\n"
#         "- Without changing the original meaning\n\n"
#         "Return only the refined question without any extra commentary."
#     )
#     payload = {
#             "contents": [{
#                 "parts": [{"text": your_question}]
#             }]
#     }

#         # Headers
#     headers = {
#             "Content-Type": "application/json"
#     }
#     valid_response = False
#     llm_refined_question=""
#     while not valid_response:
#         try:
#             response = requests.post(GEMINI_URL + GEMINI_API_KEY[increment_counter() % g_l], headers=headers,
#                                          data=json.dumps(payload))
#             llm_refined_question = response.json()["candidates"][0]["content"]["parts"][0]["text"]
#             valid_response = True
#         except Exception as e:
#             print("found exception: ", e)
#             valid_response = False

#     results = system.get_similar_chunks(llm_refined_question, "EOC_1", limit=15)
#     print(results)
#     text=""
#     for i in results:
#         temp=i["text"]
#         text1=get_summaries_by_subcategory_id(temp)
#         # print(text1)
#         text+=text1
#         # exit(1)

#     your_question = (
#         "You are a highly intelligent assistant. Based on the provided context text and the user question, "
#         "analyze the information thoroughly and extract the best possible answer. Use logical reasoning and "
#         "stay strictly within the given context.\n\n"
#         "Context Text:\n"
#         f"{text}\n\n"
#         "User Question:\n"
#         f"{llm_refined_question}\n\n"
#         "Instructions:\n"
#         "- answer like an helpful agent insure will answer."
#         "- Dont miss any points answer detailer if u feel need for certain summery or detail oriented questions"
#         "- Analyze the context deeply.\n"
#         "- Think logically and precisely.\n"
#         "- Extract and formulate the most accurate and concise answer possible.\n"
#         "- Do not make assumptions or add information that is not found in the context.\n\n"
#         "Answer:"
#     )

#     payload = {
#         "contents": [{
#             "parts": [{"text": your_question}]
#         }]
#     }

#     # Headers
#     headers = {
#         "Content-Type": "application/json"
#     }
#     valid_response = False
#     final_answer = ""
#     while not valid_response:
#         try:
#             response = requests.post(GEMINI_URL + GEMINI_API_KEY[increment_counter() % g_l], headers=headers,
#                                      data=json.dumps(payload))
#             final_answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
#             valid_response = True
#         except Exception as e:
#             print("found exception: ", e)
#             valid_response = False

#     print(final_answer)






def analyze(query: str, doc_name: str) -> str:
    system = DocumentEmbeddingSystem()

    refined_prompt = (
        f'The user has asked a question: "{query}"\n\n'
        "Please improve and refine the question to make it:\n"
        "- Clearer\n"
        "- More detailed if needed\n"
        "- Grammatically correct\n"
        "- Without changing the original meaning\n\n"
        "Return only the refined question without any extra commentary."
    )

    payload = {"contents": [{"parts": [{"text": refined_prompt}]}]}
    headers = {"Content-Type": "application/json"}

    valid_response = False
    refined_query = ""
    while not valid_response:
        try:
            response = requests.post(GEMINI_URL + GEMINI_API_KEY[increment_counter() % g_l],
                                     headers=headers, data=json.dumps(payload))
            refined_query = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            valid_response = True
        except Exception as e:
            print("Gemini refine error:", e)

    results = system.get_similar_chunks(refined_query, doc_name, limit=15)

    full_context = ""
    for result in results:
        if "text" in result:
            summary = get_summaries_by_subcategory_id(result["text"])
            if summary:
                full_context += summary

    final_prompt = (
        "You are a highly intelligent assistant. Based on the provided context text and the user question, "
        "analyze the information thoroughly and extract the best possible answer. Use logical reasoning and "
        "stay strictly within the given context.\n\n"
        f"Context Text:\n{full_context}\n\n"
        f"User Question:\n{refined_query}\n\n"
        "Instructions:\n"
        "- Answer like a helpful insurance agent would.\n"
        "- Do not miss important points.\n"
        "- Do not add anything not in the context.\n\n"
        "Answer:"
    )

    payload["contents"][0]["parts"][0]["text"] = final_prompt

    valid_response = False
    final_answer = ""
    while not valid_response:
        try:
            response = requests.post(GEMINI_URL + GEMINI_API_KEY[increment_counter() % g_l],
                                     headers=headers, data=json.dumps(payload))
            final_answer = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            valid_response = True
        except Exception as e:
            print("Gemini answer error:", e)

    return final_answer

result = analyze("how do mice behave under space conditions", "pone.pdf")
print("\n🧾 Final Answer from Gemini:\n", result)