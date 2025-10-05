import json
import sys
import threading
from imghdr import tests
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid
import openai
from neo4j import GraphDatabase

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
TOKEN_LIMIT = 4000
MAX_WORKERS = 1

GEMINI_API_KEY = ["AIzaSyBByU1sw8j2YoJbvVZ7hOATjyDTBRpVLSA"]
g_l = len(GEMINI_API_KEY)
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key="

URI = "bolt://neo4j-pb.eastus.azurecontainer.io:7687"
USERNAME = "neo4j"
PASSWORD = "test1234"

OPENAI_API_KEY = "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"


# ---------------------------------------------------------------------
# QDRANT + EMBEDDING SYSTEM
# ---------------------------------------------------------------------
class DocumentEmbeddingSystem:
    def __init__(self):
        self.client = QdrantClient(url="http://qdrant-pb.eastus.azurecontainer.io:6333")
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.vector_size = self.model.get_sentence_embedding_dimension()

    def create_collections(self):
        collection_name = "document_chunks"
        collections = self.client.get_collections()
        collection_exists = collection_name in [c.name for c in collections.collections]

        if collection_exists:
            collection_info = self.client.get_collection(collection_name)
            existing_vector_size = collection_info.config.params.vectors.size
            if existing_vector_size != self.vector_size:
                self.client.delete_collection(collection_name)
                collection_exists = False

        if not collection_exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
            )
        return collection_name

    def add_chunk(self, document_id, chunk_text, chunk_index=None, metadata=None):
        collection_name = self.create_collections()
        if chunk_index is None:
            filter_param = models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            )
            existing_chunks = self.client.scroll(collection_name=collection_name, scroll_filter=filter_param, limit=10000)
            if existing_chunks[0]:
                max_index = max(point.payload.get("chunk_index", -1) for point in existing_chunks[0])
                chunk_index = max_index + 1
            else:
                chunk_index = 0

        embedding = self.model.encode(chunk_text)
        if len(embedding) != self.vector_size:
            raise ValueError(f"Expected embedding dimension {self.vector_size}, but got {len(embedding)}")

        embedding = embedding.tolist()
        point_id = str(uuid.uuid4())

        payload = {
            "document_id": document_id,
            "chunk_index": chunk_index,
            "text": chunk_text,
            "point_id": point_id
        }
        if metadata and isinstance(metadata, dict):
            payload.update(metadata)

        point = models.PointStruct(id=point_id, vector=embedding, payload=payload)
        try:
            self.client.upsert(collection_name=collection_name, points=[point])
            return point_id
        except Exception as e:
            print(f"Error upserting point: {e}")
            raise

    def get_similar_chunks(self, query_text, document_id=None, limit=5):
        collection_name = "document_chunks"
        query_vector = self.model.encode(query_text).tolist()
        filter_param = None
        if document_id:
            filter_param = models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            )

        search_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=filter_param
        )

        results = []
        for res in search_results:
            result_data = {"point_id": res.id, "similarity_score": res.score}
            result_data.update(res.payload)
            results.append(result_data)
        return results


# ---------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------
counter_lock = threading.Lock()
COUNT = 0

def increment_counter():
    global COUNT
    with counter_lock:
        current = COUNT
        COUNT += 1
        return current


def get_summaries_by_subcategory_id(subcategory_id):
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        query = """
        MATCH (s:SubAttribute {id: $subcategory_id})
        OPTIONAL MATCH (s)<-[:DESCRIBES]-(t:TextSummary)
        WITH s, collect(t) as summaries
        RETURN s.id as subcategory_id, s.name as subcategory_name, summaries
        """
        result = session.run(query, subcategory_id=subcategory_id)
        record = result.single()
        if not record:
            return None

        summary_list = ""
        for summary in record["summaries"]:
            if summary is not None and "content" in summary:
                summary_list += summary["content"] + "\n"
        return summary_list


def refine_question_with_llm(user_input):
    your_question = (
        f'The user has asked a question: "{user_input}"\n\n'
        "Please improve and refine the question to make it clearer and more detailed without changing meaning.\n"
        "Return only the refined question."
    )
    payload = {"contents": [{"parts": [{"text": your_question}]}]}
    headers = {"Content-Type": "application/json"}
    valid_response = False
    llm_refined_question = ""
    while not valid_response:
        try:
            response = requests.post(
                GEMINI_URL + GEMINI_API_KEY[increment_counter() % g_l],
                headers=headers,
                data=json.dumps(payload)
            )
            llm_refined_question = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            valid_response = True
        except Exception as e:
            print("Error refining question:", e)
            valid_response = False
    return llm_refined_question


def are_all_uuids(items):
    for item in items:
        try:
            uuid_obj = uuid.UUID(str(item))
            if str(uuid_obj) != item:
                return False
        except Exception:
            return False
    return True


def get_relevant_node_ids(json_structure, refined_question):
    your_question = (
        "You are given a structured list of attributes with IDs:\n"
        f"{json.dumps(json_structure)}\n\n"
        "Find only sub-attributes relevant (>70%) to the user question below.\n"
        f"USER QUESTION:\n{refined_question}\n\n"
        "Return valid JSON list of node IDs only."
    )
    openai.api_key = OPENAI_API_KEY
    validate_uuids = False
    node_ids = {}
    while not validate_uuids:
        try:
            response = openai.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": your_question}
                ]
            )
            content = response.choices[0].message.content
            print("   relevant node ids   ")
            print(content)
            node_ids = json.loads(content)
            validate_uuids = are_all_uuids(node_ids)
        except Exception as e:
            print(f"Error loading JSON files: {e}")
            validate_uuids = False
    return node_ids


# ---------------------------------------------------------------------
# COMPARISON + RECOMMENDATION
# ---------------------------------------------------------------------
def recommend_documents(comparison_json, refined_question):
    recommendation_prompt = (
        "Analyze the structured JSON comparison and user question to recommend top research papers.\n"
        f"User Question:\n{refined_question}\n\nComparison JSON:\n{comparison_json}\n\n"
        "Return valid JSON with papers having >75% match."
    )
    openai.api_key = OPENAI_API_KEY
    response = openai.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": recommendation_prompt}
        ]
    )
    return response.choices[0].message.content


def compare_documents(file_texts, refined_question):
    your_question = (
        "Compare these documents in structured JSON form based only on the given text.\n\n"
    )
    for doc_dict in file_texts:
        for doc_id, doc_content in doc_dict.items():
            your_question += f"{doc_id}:\n{doc_content}\n\n"
    your_question += f"User Question:\n{refined_question}\n\nAnswer in valid JSON."

    openai.api_key = OPENAI_API_KEY
    response = openai.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": your_question}
        ]
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------
with open("/Users/pranaligole/MS/Courses/Sem2/CS532/Code/Final/Agent/augmented_documents.json", "r") as doc_f:
    aug_doc = json.load(doc_f)
    ls_documents = aug_doc["DOCUMENTS"]


def recommend(user_input):
    print(ls_documents)
    system = DocumentEmbeddingSystem()
    llm_refined_question = refine_question_with_llm(user_input)
    print(f"Refined question: {llm_refined_question}")
    tests = []
    MAX_TEXT_LENGTH = 8000

    for i in ls_documents:
        try:
            with open(f"{i}.json", "r") as f1:
                file_1_json = json.load(f1)
        except FileNotFoundError as e:
            print(f"Error loading JSON files: {e}")
            return

        results1 = system.get_similar_chunks(llm_refined_question, i, limit=15)

        if results1:
            avg_score = np.mean([r["similarity_score"] for r in results1])
            print(f"📄 Document: {i} | Average Similarity: {avg_score:.3f}")
            if avg_score >= 0.75:
                print(f"✅ Highly relevant: {i}")
        else:
            print(f"⚠️ No relevant chunks found for {i}")

        text1 = ""
        for jr in results1:
            temp = jr["text"]
            temp1 = get_summaries_by_subcategory_id(temp)
            if temp1:
                text1 += temp1
            if len(text1) > MAX_TEXT_LENGTH:
                break

        node_ids1 = get_relevant_node_ids(file_1_json, llm_refined_question)
        for jn in node_ids1:
            temp1 = get_summaries_by_subcategory_id(jn)
            if temp1:
                text1 += temp1
            if len(text1) > MAX_TEXT_LENGTH:
                break

        tests.append({i: text1[:MAX_TEXT_LENGTH]})

    comparison_result = compare_documents(tests, llm_refined_question)
    recomendation = recommend_documents(comparison_result, llm_refined_question)

    try:
        rec_obj = json.loads(recomendation) if isinstance(recomendation, str) else recomendation
        if isinstance(rec_obj, dict) and "Recommendations" in rec_obj:
            parts = []
            for rec in rec_obj["Recommendations"]:
                parts.append(f"### {rec['Document']}\n- {rec['Reasoning']}\n")
            return "\n".join(parts)
    except Exception:
        pass

    print("\n==== FINAL RANKED RESEARCH PAPERS ====")
    for doc_id in ls_documents:
        results = system.get_similar_chunks(llm_refined_question, doc_id, limit=10)
        if results:
            avg_score = np.mean([r["similarity_score"] for r in results])
            if avg_score >= 0.75:
                print(f"⭐ {doc_id} → {avg_score:.2f}")

    top_papers = []
    for doc_id in ls_documents:
        results = system.get_similar_chunks(llm_refined_question, doc_id, limit=10)
        if results:
            avg_score = np.mean([r["similarity_score"] for r in results])
            if avg_score >= 0.75:
                top_papers.append({"document": doc_id, "avg_score": avg_score})

    with open("/Users/pranaligole/MS/Courses/Sem2/CS532/Code/Data/top.json", "w") as f:
        json.dump(top_papers, f, indent=2)
    print("\n📂 Saved top papers → top.json")

    print("\nRecommendation Result:")
    return recomendation


recommend("What did the Bion-M1 mice training and selection study reveal about how preflight conditioning influences physiological adaptation to microgravity?")
