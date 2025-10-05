# import json
# import sys
# import threading
# import time
# import uuid
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from datetime import datetime
# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import PyPDF2
# import requests

# URI = "bolt://neo4j-pb.eastus.azurecontainer.io:7687"
# USERNAME = "neo4j"
# PASSWORD = "test1234"

# TOKEN_LIMIT=4000
# MAX_WORKERS = 6
# MAX_WORKERS_EMBEDDINGS = 6
# from neo4j import GraphDatabase
# GEMINI_API_KEY = ["AIzaSyBByU1sw8j2YoJbvVZ7hOATjyDTBRpVLSA"]  # Replace with your actual API key
# g_l=len(GEMINI_API_KEY)
# GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key="

# # def query_groq(g_prompt, groq_model):
# #     client = Groq(api_key=GROQ_API_KEY[increment_counter() % token_l])
# #     chat_completion = client.chat.completions.create(
# #         messages=[{"role": "user", "content": g_prompt}],
# #         model=groq_model)
# #     return chat_completion.choices[0].message.content

# # with open("data_1.json", "r") as file_1:
# #     data_attributes=json.load(file_1)
# #
# # with open("main_attribute_data.json_1","r") as file_2:
# #     data_main_attributes=json.load(file_2)

# data_attributes={}
# data_main_attributes={}


# class DocumentEmbeddingSystem:
#     def __init__(self):
#         self.client = QdrantClient(url="http://qdrant-pb.eastus.azurecontainer.io:6333")
#         self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
#         self.vector_size = self.model.get_sentence_embedding_dimension()

#     def create_collections(self):
#         """Create a single collection for all documents"""
#         collection_name = "document_chunks"

#         # Check if collection exists
#         collections = self.client.get_collections()
#         collection_exists = collection_name in [c.name for c in collections.collections]

#         if collection_exists:
#             # If it exists, check if it has the correct vector size
#             collection_info = self.client.get_collection(collection_name)
#             existing_vector_size = collection_info.config.params.vectors.size

#             if existing_vector_size != self.vector_size:
#                 # If sizes don't match, recreate the collection
#                 print(f"Vector size mismatch: collection has {existing_vector_size}, model outputs {self.vector_size}")
#                 print(f"Recreating collection {collection_name}...")
#                 self.client.delete_collection(collection_name)
#                 collection_exists = False

#         if not collection_exists:
#             # Create collection with the correct vector size
#             self.client.create_collection(
#                 collection_name=collection_name,
#                 vectors_config=models.VectorParams(
#                     size=self.vector_size,
#                     distance=models.Distance.COSINE
#                 )
#             )
#             print(f"Created collection '{collection_name}' with vector size {self.vector_size}")

#         return collection_name

#     def add_chunk(self, document_id, chunk_text, node_id, chunk_index=None, metadata=None):
#         """
#         Add a single chunk to a document

#         Args:
#             document_id: Identifier for the document
#             chunk_text: Text content of the chunk
#             chunk_index: Optional index/position of the chunk in the document
#             metadata: Optional additional metadata as a dictionary

#         Returns:
#             The point ID stored in Qdrant
#         """
#         collection_name = self.create_collections()

#         # If chunk_index not provided, try to determine next index
#         if chunk_index is None:
#             # Get current chunks for this document
#             filter_param = models.Filter(
#                 must=[
#                     models.FieldCondition(
#                         key="document_id",
#                         match=models.MatchValue(value=document_id)
#                     )
#                 ]
#             )

#             existing_chunks = self.client.scroll(
#                 collection_name=collection_name,
#                 scroll_filter=filter_param,
#                 limit=10000
#             )

#             if existing_chunks[0]:
#                 # Find highest chunk_index and add 1
#                 max_index = max(point.payload.get("chunk_index", -1) for point in existing_chunks[0])
#                 chunk_index = max_index + 1
#             else:
#                 # No existing chunks, start at 0
#                 chunk_index = 0

#         # Generate embedding for the chunk
#         embedding = self.model.encode(chunk_text)

#         # Verify embedding dimension
#         if len(embedding) != self.vector_size:
#             raise ValueError(f"Expected embedding dimension {self.vector_size}, but got {len(embedding)}")

#         # Convert to list for JSON serialization
#         embedding = embedding.tolist()

#         # Generate unique ID
#         point_id = str(uuid.uuid4())

#         # Prepare payload
#         payload = {
#             "document_id": document_id,
#             "chunk_index": chunk_index,
#             "text": node_id,
#             "point_id": point_id
#         }

#         # Add custom metadata if provided
#         if metadata and isinstance(metadata, dict):
#             payload.update(metadata)

#         # Create and store the point
#         point = models.PointStruct(
#             id=point_id,
#             vector=embedding,
#             payload=payload
#         )

#         try:
#             self.client.upsert(
#                 collection_name=collection_name,
#                 points=[point]
#             )
#             return point_id
#         except Exception as e:
#             print(f"Error upserting point: {e}")
#             print(f"Vector dimension: {len(embedding)}")
#             raise

#     def get_similar_chunks(self, query_text, document_id=None, limit=5):
#         """
#         Find chunks similar to the query text

#         Args:
#             query_text: The query to find similar chunks for
#             document_id: Optional - filter results to a specific document
#             limit: Maximum number of results to return

#         Returns:
#             List of similar chunks with metadata
#         """
#         collection_name = "document_chunks"

#         # Generate embedding for query
#         query_vector = self.model.encode(query_text).tolist()

#         # Prepare filter if document_id is specified
#         filter_param = None
#         if document_id:
#             filter_param = models.Filter(
#                 must=[
#                     models.FieldCondition(
#                         key="document_id",
#                         match=models.MatchValue(value=document_id)
#                     )
#                 ]
#             )

#         # Search in Qdrant
#         search_results = self.client.search(
#             collection_name=collection_name,
#             query_vector=query_vector,
#             limit=limit,
#             query_filter=filter_param
#         )

#         # Format results
#         results = []
#         for res in search_results:
#             # Extract all payload data
#             result_data = {
#                 "point_id": res.id,
#                 "similarity_score": res.score
#             }
#             # Add all payload fields
#             result_data.update(res.payload)
#             results.append(result_data)

#         return results

#     def get_document_chunks(self, document_id):
#         """
#         Retrieve all chunks for a specific document

#         Args:
#             document_id: ID of the document to retrieve chunks for

#         Returns:
#             List of chunks in order of their chunk_index
#         """
#         collection_name = "document_chunks"

#         # Create filter for document_id
#         filter_param = models.Filter(
#             must=[
#                 models.FieldCondition(
#                     key="document_id",
#                     match=models.MatchValue(value=document_id)
#                 )
#             ]
#         )

#         # Query Qdrant
#         scroll_results = self.client.scroll(
#             collection_name=collection_name,
#             scroll_filter=filter_param,
#             limit=10000  # Adjust based on your expected document size
#         )

#         # Extract chunks with all metadata
#         chunks = []
#         for point in scroll_results[0]:
#             chunk_data = {
#                 "point_id": point.id
#             }
#             chunk_data.update(point.payload)
#             chunks.append(chunk_data)

#         # Sort by chunk_index
#         chunks.sort(key=lambda x: x["chunk_index"])

#         return chunks

#     def count_document_chunks(self, document_id):
#         """Count the number of chunks in a document"""
#         collection_name = "document_chunks"

#         filter_param = models.Filter(
#             must=[
#                 models.FieldCondition(
#                     key="document_id",
#                     match=models.MatchValue(value=document_id)
#                 )
#             ]
#         )

#         count = self.client.count(
#             collection_name=collection_name,
#             count_filter=filter_param
#         )

#         return count.count


# def setup_neo4j_constraints():
#     driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
#     with driver.session() as session:
#         # Create unique constraints
#         try:
#             # Make sure SubAttribute names are unique under a MainAttribute
#             session.run(
#                 "CREATE CONSTRAINT unique_sub_attr_name IF NOT EXISTS "
#                 "FOR (s:SubAttribute) "
#                 "REQUIRE (s.name, s.belongs_to_main) IS UNIQUE"
#             )

#             # Make sure TextSummary content is unique for a SubAttribute
#             session.run(
#                 "CREATE CONSTRAINT unique_text_summary IF NOT EXISTS "
#                 "FOR (t:TextSummary) "
#                 "REQUIRE (t.content, t.describes_sub) IS UNIQUE"
#             )

#             # Make sure MainAttribute names are unique under a Document
#             session.run(
#                 "CREATE CONSTRAINT unique_main_attr_name IF NOT EXISTS "
#                 "FOR (m:MainAttribute) "
#                 "REQUIRE (m.name, m.part_of_doc) IS UNIQUE"
#             )

#         except Exception as e:
#             print(f"Error setting up constraints: {e}")

# def create_document_node(doc_name, properties=None):
#     """
#     Create a Document node to represent the PDF file

#     Args:
#         doc_name (str): Name of the document
#         properties (dict, optional): Additional properties for the node

#     Returns:
#         str: ID of the created Document node
#     """
#     node_id = str(uuid.uuid4())
#     driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
#     with driver.session() as session:
#         # Create document node
#         cypher_query = (
#             "CREATE (d:Document {id: $id, name: $name}) "
#             "RETURN d"
#         )

#         params = {"id": node_id, "name": doc_name}

#         # Add any additional properties
#         if properties:
#             property_set = ", ".join([f"d.{key} = ${key}" for key in properties.keys()])
#             cypher_query = (
#                 f"CREATE (d:Document {{id: $id, name: $name}}) "
#                 f"SET {property_set} "
#                 f"RETURN d"
#             )
#             params.update(properties)

#         session.run(cypher_query, params)

#     return node_id
# # def create_main_attribute(name, properties=None):
# #         """
# #         Create a main attribute node
# #
# #         Args:
# #             name (str): Name of the main attribute
# #             properties (dict, optional): Additional properties for the node
# #
# #         Returns:
# #             str: ID of the created node
# #         """
# #         node_id = str(uuid.uuid4())
# #         driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
# #         with driver.session() as session:
# #             # Create main attribute node
# #             cypher_query = (
# #                 "CREATE (m:MainAttribute {id: $id, name: $name}) "
# #                 "RETURN m"
# #             )
# #
# #             params = {"id": node_id, "name": name}
# #
# #             # Add any additional properties
# #             if properties:
# #                 property_set = ", ".join([f"m.{key} = ${key}" for key in properties.keys()])
# #                 cypher_query = (
# #                     f"CREATE (m:MainAttribute {{id: $id, name: $name}}) "
# #                     f"SET {property_set} "
# #                     f"RETURN m"
# #                 )
# #                 params.update(properties)
# #
# #             session.run(cypher_query, params)
# #
# #         return node_id

# def create_main_attribute(name, document_id, properties=None):
#     """
#     Create a main attribute node connected to a document

#     Args:
#         name (str): Name of the main attribute
#         document_id (str): ID of the document this attribute belongs to
#         properties (dict, optional): Additional properties for the node

#     Returns:
#         str: ID of the created node
#     """
#     node_id = str(uuid.uuid4())
#     driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
#     with driver.session() as session:
#         # Create main attribute node and connect to document
#         cypher_query = (
#             "MATCH (d:Document {id: $doc_id}) "
#             "CREATE (m:MainAttribute {id: $id, name: $name})-[:PART_OF]->(d) "
#             "RETURN m"
#         )

#         params = {"id": node_id, "name": name, "doc_id": document_id}

#         # Add any additional properties
#         if properties:
#             property_set = ", ".join([f"m.{key} = ${key}" for key in properties.keys()])
#             cypher_query = (
#                 "MATCH (d:Document {id: $doc_id}) "
#                 "CREATE (m:MainAttribute {id: $id, name: $name})-[:PART_OF]->(d) "
#                 f"SET {property_set} "
#                 "RETURN m"
#             )
#             params.update(properties)

#         session.run(cypher_query, params)

#     return node_id


# def create_sub_attribute(main_attr_id, name, properties=None):
#     """
#     Create a sub-attribute node connected to a main attribute

#     Args:
#         main_attr_id (str): ID of the main attribute
#         name (str): Name of the sub-attribute
#         properties (dict, optional): Additional properties for the node

#     Returns:
#         str: ID of the created sub-attribute node
#     """
#     node_id = str(uuid.uuid4())
#     driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
#     with driver.session() as session:
#         # Create sub-attribute and connect to main attribute
#         cypher_query = (
#             "MATCH (m:MainAttribute {id: $main_id}) "
#             "CREATE (s:SubAttribute {id: $id, name: $name})-[:BELONGS_TO]->(m) "
#             "RETURN s"
#         )

#         params = {"main_id": main_attr_id, "id": node_id, "name": name}

#         # Add any additional properties
#         if properties:
#             property_set = ", ".join([f"s.{key} = ${key}" for key in properties.keys()])
#             cypher_query = (
#                 "MATCH (m:MainAttribute {id: $main_id}) "
#                 "CREATE (s:SubAttribute {id: $id, name: $name})-[:BELONGS_TO]->(m) "
#                 f"SET {property_set} "
#                 "RETURN s"
#             )
#             params.update(properties)

#         session.run(cypher_query, params)

#     return node_id


# def add_text_summary_with_merge(sub_attr_id, summary_name, text_content, properties=None):
#     node_id = str(uuid.uuid4())
#     driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

#     # Validate UUID
#     try:
#         uuid.UUID(sub_attr_id)
#     except ValueError:
#         print(f"Value of '{sub_attr_id}' is not a valid UUID:")
#         return ""

#     with driver.session() as session:
#         # Use MERGE to find or create the summary
#         cypher_query = (
#             "MATCH (s:SubAttribute {id: $sub_id}) "
#             "MERGE (t:TextSummary {content: $content})-[:DESCRIBES]->(s) "
#             "ON CREATE SET t.id = $id, t.name = $name "
#             "RETURN t.id as summary_id"
#         )

#         params = {
#             "sub_id": sub_attr_id,
#             "id": node_id,
#             "name": summary_name,
#             "content": text_content
#         }

#         # Add any additional properties
#         if properties:
#             property_set = ", ".join([f"t.{key} = ${key}" for key in properties.keys()])
#             cypher_query = (
#                 "MATCH (s:SubAttribute {id: $sub_id}) "
#                 "MERGE (t:TextSummary {content: $content})-[:DESCRIBES]->(s) "
#                 "ON CREATE SET t.id = $id, t.name = $name "
#                 f"ON CREATE SET {property_set} "
#                 "RETURN t.id as summary_id"
#             )
#             params.update(properties)

#         result = session.run(cypher_query, params)
#         record = result.single()
#         return record["summary_id"] if record else None

# # def add_text_summary( sub_attr_id, summary_name, text_content, properties=None):
# #     node_id = str(uuid.uuid4())
# #     driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
# #     try:
# #
# #         # Check if the string is a valid UUID
# #         uuid.UUID(sub_attr_id)
# #     except ValueError:
# #         print( f"Value of '{sub_attr_id}' is not a valid UUID:")
# #         return ""
# #
# #     with driver.session() as session:
# #         # Create text summary and connect to sub-attribute
# #         cypher_query = (
# #             "MATCH (s:SubAttribute {id: $sub_id}) "
# #             "CREATE (t:TextSummary {id: $id, name: $name, content: $content})-[:DESCRIBES]->(s) "
# #             "RETURN t"
# #         )
# #
# #         params = {
# #             "sub_id": sub_attr_id,
# #             "id": node_id,
# #             "name": summary_name,
# #             "content": text_content
# #         }
# #
# #         # Add any additional properties
# #         if properties:
# #             property_set = ", ".join([f"t.{key} = ${key}" for key in properties.keys()])
# #             cypher_query = (
# #                 "MATCH (s:SubAttribute {id: $sub_id}) "
# #                 "CREATE (t:TextSummary {id: $id, name: $name, content: $content})-[:DESCRIBES]->(s) "
# #                 f"SET {property_set} "
# #                 "RETURN t"
# #             )
# #             params.update(properties)
# #
# #         session.run(cypher_query, params)
# #
# #     return node_id


# # Connect to Neo4j and insert data
# def insert_data(jsondata1, title, doc_id):
#     description=""
#     driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
#     with driver.session() as session:
#         for category, terms in jsondata1.items():
#             if category not in data_attributes:
#                 mainattr_id=create_main_attribute(category, doc_id,{"created_date": str(datetime.now())})
#                 data_main_attributes[category]=mainattr_id
#                 data_attributes[category]={}
#             if isinstance(terms, str):
#                 description=terms
#                 add_text_summary_with_merge( data_main_attributes[category], title, description,{"created_date": str(datetime.now())})
#             else:
#                 for term, desc in terms.items():
#                     if data_attributes[category]=={} or (term not in data_attributes[category]):
#                         mainattr_id=data_main_attributes[category]
#                         subattr_id=create_sub_attribute(mainattr_id, term, {"created_date": str(datetime.now())})
#                         data_attributes[category][term]=subattr_id
#                     add_text_summary_with_merge(data_attributes[category][term], title, desc, {"created_date": str(datetime.now())})
#     print("Nodes and relationships created successfully!")
#     driver.close()

# # def section_execution(data, task_id_se, text_length, title):
# #     # summery=groq_summary(data, g_llm, text_length)
# #     relation=groq_relations(data, data_attributes, g_llm, text_length)
# #     insert_data(relation, title)

# # def process_section(section_1, task_id_1):
# #     l = len(section_1)
# #     summary_title = "summery_"+str(task_id_1)
# #     result=section_execution(section_1, task_id_1, l, summary_title)
# #     # print(f"Task {task_id_1} completed.")
# #     return result


# def validate_json_structure(json_str_val):
#     try:
#         # Parse the JSON string
#         data = json.loads(json_str_val)

#         # Check if there's at least one top-level attribute
#         if len(data) < 1:
#             return True, "JSON should have at least one top-level attribute"

#         # Iterate through all top-level attributes
#         for main_attr_name, main_attr in data.items():
#             # Check if main attribute is an object
#             if not isinstance(main_attr, dict):
#                 return False, f"The main attribute '{main_attr_name}' should be an object"

#             # Check if all values in the main attribute are UUID strings
#             for sub_attr, value in main_attr.items():
#                 # Check if the value is a string
#                 if not isinstance(value, str):
#                     return False, f"Value of '{sub_attr}' in '{main_attr_name}' is not a string"

#         return True, "JSON structure is valid"

#     except json.JSONDecodeError as e:
#         return False, f"Invalid JSON: {e}"
#     except Exception as e:
#         return False, f"Validation error: {e}"


# def process_response(response_1):
#     # print("line 340")
#     # print(response_1)
#     tag = "</think>"
#     text = ""
#     tag_position = response_1.find(tag)
#     after=""
#     # If the tag exists, return everything after the tag
#     if tag_position != -1:
#         text = response_1[tag_position + len(tag):]
#     else:
#         # If tag not found, return the original text
#         text = response_1
#     if text.find("{"):
#         before, after = text.split("{", 1)
#     after = "{" + after
#     # with open("final_one_11.txt", "a") as file_3:
#     #     file_3.write(after)

#     after_rr = after.rstrip('`').strip()
#     # print("after strip response:    ")
#     # print(after_rr)
#     # print("\n")
#     valid_response_1, validation_reason = validate_json_structure(after_rr)
#     if not valid_response_1:
#         repeat_boolean_1 = True
#     else:
#         repeat_boolean_1 = False
#     return after_rr, repeat_boolean_1, valid_response_1

# counter_lock = threading.Lock()
# COUNT=0
# def increment_counter():
#     global COUNT
#     with counter_lock:
#         current = COUNT
#         COUNT += 1
#         return current


# def get_summaries_by_subcategory_id(subcategory_id):
#     driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
#     with driver.session() as session:
#         query = """
#         MATCH (s:SubAttribute {id: $subcategory_id})
#         OPTIONAL MATCH (s)<-[:DESCRIBES]-(t:TextSummary)
#         WITH s, collect(t) as summaries
#         RETURN s.id as subcategory_id, 
#                s.name as subcategory_name, 
#                s.priority as priority,
#                summaries
#         """
#         result = session.run(query, subcategory_id=subcategory_id)
#         record = result.single()
#         # print(record)
#         if not record:
#             return None

#         word_count = 0
#         summary_list = ""

#         # Debug the record
#         print(f"SubAttribute: {record['subcategory_name']}")

#         # The issue is in how you're accessing the summaries
#         for summary in record["summaries"]:
#             if summary is not None:
#                 # In Neo4j, the nodes are returned as Neo4j node objects
#                 # You need to access their properties using dictionary syntax
#                 if "content" in summary:
#                     content = summary["content"]
#                     summary_list += content + "\n"
#                     word_count += len(content.split())

#         print("word count: " + str(word_count))
#         return [word_count, summary_list]

# def process_text_from_pdf(text, page_num, doc_id):
#     valid_response = False
#     after_process_text=""
#     repeat_boolean = True
#     global response
#     while repeat_boolean:
#         # your_question = (
#         #     "Given the following list of attributes and sub-attributes:\n"
#         #     f"{json.dumps(data_attributes)}\n\n"
#         #     "Your task is to analyze the text below and map it to the most relevant main attributes and sub-attributes.\n\n"
#         #     "Instructions:\n"
#         #     "1. STRICTLY select from the provided list whenever possible.\n"
#         #     "2. If no suitable attribute exists, you are allowed to create a new attribute and logical sub-attributes.\n"
#         #     "3. If creating a new attribute, choose a meaningful, intuitive name based on the concepts in the text — do NOT use generic names like 'NEW_ATTRIBUTE'.\n"
#         #     "4. For each sub-attribute matched, extract a short, precise description from the text.\n"
#         #     "5. Return ONLY a JSON object in the following format:\n"
#         #     "{\n"
#         #     "  \"<main_attribute>\": {\n"
#         #     "    \"<sub_attribute_1>\": \"<description>\",\n"
#         #     "    \"<sub_attribute_2>\": \"<description>\"\n"
#         #     "  }\n"
#         #     "}\n\n"
#         #     f"TEXT TO ANALYZE: {text}\n\n"
#         #     "IMPORTANT:\n"
#         #     "- When matching an attribute, the relevance must be above 70%.\n"
#         #     "- If no suitable match exists, create a logical, clear new attribute.\n"
#         #     "- Attribute and sub-attribute names must be meaningful and derived from the text.\n"
#         #     "- Return ONLY valid JSON with no extra text, no explanations.\n"
#         #     "- If nothing matches, return an empty JSON object: {}"
#         # )
#         your_question = (
#             "Given the following list of main attributes and their sub-attributes:\n"
#             f"{json.dumps(data_attributes)}\n\n"

#             "Your task is to analyze the entire text below and comprehensively map it to the most relevant main attributes and sub-attributes from the list.\n\n"

#             "Instructions:\n"
#             "1. ONLY select from the provided attribute list when a relevant match is found in the text.\n"
#             "2. If no suitable match exists, create a new main attribute and clear, meaningful sub-attributes — avoid placeholders like 'NEW_ATTRIBUTE'.\n"
#             "3. You MUST identify and return all relevant sub-attributes — do not stop at the first match.\n"
#             "4. For each sub-attribute, extract a short but specific description or excerpt from the original text that directly supports the match.\n"
#             "   - Do NOT summarize vaguely. Include all essential facts, clauses, or phrases that explain the sub-attribute's relevance.\n"
#             "   - If multiple relevant details exist in the text, merge them logically into a single description.\n"
#             "   - Do not miss or skip over any informative parts tied to the sub-attribute.\n"
#             "5. Ignore content that does not meaningfully relate to the attribute list.\n"
#             "6. Return ONLY a valid JSON object in this format — no explanation, no additional text:\n"
#             "{\n"
#             "  \"<main_attribute>\": {\n"
#             "    \"<sub_attribute_1>\": \"<relevant description from text>\",\n"
#             "    \"<sub_attribute_2>\": \"<relevant description from text>\"\n"
#             "  },\n"
#             "  ... (more attributes)\n"
#             "}\n\n"

#             f"TEXT TO ANALYZE:\n{text}\n\n"

#             "IMPORTANT:\n"
#             "- Be exhaustive: extract all meaningful matches across the entire text.\n"
#             "- Relevance threshold for matching: above 70% semantic similarity.\n"
#             "- For any unmatched but meaningful concept, create a new attribute and sub-attributes.\n"
#             "- The result must be clean, structured, and strictly JSON formatted — no commentary.\n"
#             "- If nothing matches, return an empty JSON object: {}"
#         )

#         payload = {
#             "contents": [{
#                 "parts": [{"text": your_question}]
#             }]
#         }

#         # Headers
#         headers = {
#             "Content-Type": "application/json"
#         }
#         try:
#             response = requests.post(GEMINI_URL + GEMINI_API_KEY[increment_counter() % g_l], headers=headers,
#                                      data=json.dumps(payload))
#             raw_text_1 = response.json()["candidates"][0]["content"]["parts"][0]["text"]
#             after_process_text, repeat_boolean, valid_response = process_response(raw_text_1)
#         except Exception as e:
#             print("found exception: ", e)
#             valid_response=False

#         if valid_response:
#             # print("VALID=========================")
#             # print(repeat_boolean)
#             # print(text)

#             insert_data(json.loads(after_process_text.replace("\n", "")), "Description_" + str(page_num), doc_id)
#         else:
#             print("INVALID json from llm ===========================================================================")
#             print(response)
#             print("count:  "+str(COUNT))
#             # repeat_boolean=True
#     return json.loads(after_process_text.replace("\n", ""))

# def extract_text_from_pdf(pdf_path, doc_name):
#     text = ""
#     document_id=create_document_node(doc_name, {"created_date": str(datetime.now())})
#     with open(pdf_path, 'rb') as pdf_file:
#         reader = PyPDF2.PdfReader(pdf_file)
#         print(len(reader.pages))
#         futures = []
#         with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#             for page_num in range(len(reader.pages)):
#                 page = reader.pages[page_num]
#                 text = page.extract_text()
#                 future = executor.submit(process_text_from_pdf, text, page_num, document_id)
#                 futures.append(future)
#             for future in as_completed(futures):
#                 r = future.result()
#                 print(r)


#         with open(f"{doc_name}.json", "w") as file_1:
#             json.dump(data_attributes, file_1)

#         with open(f"main_attribute_{doc_name}.json", "w") as file_2:
#             json.dump(data_main_attributes, file_2)
#             # exit(1)
#             # break
#     return text


# def execute_embeddings_from_text(system, node_id, doc_id, category, sub_category):
#     # Get summaries for the specific node
#     summaries = get_summaries_by_subcategory_id(node_id)
#     final_summery=(f"Category: {category}"
#         f"Subcategory: {sub_category}"
#         f"Summary: {summaries[1]}")
#     # Check if summaries exist
#     if not summaries or len(summaries) < 2 or not summaries[1]:
#         return f"No valid text content found for node {node_id}"

#     try:
#         # Add the text as a chunk to the embedding system
#         chunk_id = system.add_chunk(
#             document_id=doc_id,
#             chunk_text=final_summery,  # Using the text content from summaries
#             node_id=node_id
#         )
#         return f"Added chunk {chunk_id} to document {doc_id} for node {node_id}"
#     except Exception as e:
#         return f"Error adding chunk for node {node_id}: {str(e)}"


# # with open(f"{doc_name_input}.json", "r") as file_1:
# #     data_attributes=json.load(file_1)

# def parse_embeddings_from_text(doc_id):
#     # Initialize statistics
#     stats = {
#         "total_nodes": 0,
#         "successful": 0,
#         "failed": 0,
#         "errors": []
#     }

#     # Create embedding system
#     try:
#         system = DocumentEmbeddingSystem()
#     except Exception as e:
#         print(f"Failed to initialize DocumentEmbeddingSystem: {str(e)}")
#         return {"error": f"Failed to initialize embedding system: {str(e)}"}

#     # Process nodes in parallel
#     futures = []
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS_EMBEDDINGS) as executor:
#         # Iterate through each main attribute and sub-attribute
#         for main, sub_attrs in data_attributes.items():
#             for sub_attr, node_id in sub_attrs.items():
#                 stats["total_nodes"] += 1
#                 future = executor.submit(execute_embeddings_from_text, system, node_id, doc_id, main, sub_attr)
#                 futures.append((future, node_id))

#         # Process results as they complete
#         for future, node_id in [(f, nid) for f, nid in futures]:
#             try:
#                 result = future.result()
#                 print(result)

#                 if "Error" in result:
#                     stats["failed"] += 1
#                     stats["errors"].append({"node_id": node_id, "error": result})
#                 else:
#                     stats["successful"] += 1
#             except Exception as e:
#                 stats["failed"] += 1
#                 error_msg = f"Exception processing node {node_id}: {str(e)}"
#                 print(error_msg)
#                 stats["errors"].append({"node_id": node_id, "error": error_msg})

#     # Return statistics about the embedding process
#     return stats
#             # exit(1)

# def run_pipeline(pdf_path, doc_name):
#     start_time = time.time()

#     # Extract and process text
#     extracted_text = extract_text_from_pdf(pdf_path, doc_name)
#     print("Extraction done. Time taken:", time.time() - start_time)

#     # Generate embeddings
#     parse_embeddings_from_text(doc_name)
#     print("Full pipeline completed. Total time:", time.time() - start_time)

#     print("time taken: "+str(time.time()-start_time))

#     return extracted_text

# def extract(pdf_path: str, doc_name: str) -> str:
#     return run_pipeline(pdf_path, doc_name)


# extract("/Users/pranaligole/MS/Courses/Sem2/CS532/Code/Data/pone.0104830.pdf", "fpls")


import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import PyPDF2
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
URI = "bolt://neo4j-pb.eastus.azurecontainer.io:7687"
USERNAME = "neo4j"
PASSWORD = "test1234"

TOKEN_LIMIT = 4000
MAX_WORKERS = 6
MAX_WORKERS_EMBEDDINGS = 6

GEMINI_API_KEY = ["AIzaSyBByU1sw8j2YoJbvVZ7hOATjyDTBRpVLSA"]
g_l = len(GEMINI_API_KEY)
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key="

data_attributes = {}
data_main_attributes = {}

# ---------------------------------------------------------
# Qdrant Embedding System
# ---------------------------------------------------------
class DocumentEmbeddingSystem:
    def __init__(self):
        self.client = QdrantClient(url="http://qdrant-pb.eastus.azurecontainer.io:6333")
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.vector_size = self.model.get_sentence_embedding_dimension()

    def create_collections(self):
        """Create/ensure the 'document_chunks' collection with correct vector size."""
        collection_name = "document_chunks"

        collections = self.client.get_collections()
        collection_exists = collection_name in [c.name for c in collections.collections]

        if collection_exists:
            collection_info = self.client.get_collection(collection_name)
            existing_vector_size = collection_info.config.params.vectors.size
            if existing_vector_size != self.vector_size:
                print(f"Vector size mismatch: collection has {existing_vector_size}, model outputs {self.vector_size}")
                print(f"Recreating collection {collection_name}...")
                self.client.delete_collection(collection_name)
                collection_exists = False

        if not collection_exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created collection '{collection_name}' with vector size {self.vector_size}")

        return collection_name

    def add_chunk(self, document_id, chunk_text, node_id, chunk_index=None, metadata=None):
        """Add a single chunk to a document."""
        collection_name = self.create_collections()

        # Determine next chunk index if not provided
        if chunk_index is None:
            filter_param = models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            )
            existing_chunks = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_param,
                limit=10000
            )
            if existing_chunks[0]:
                max_index = max(point.payload.get("chunk_index", -1) for point in existing_chunks[0])
                chunk_index = max_index + 1
            else:
                chunk_index = 0

        # Embed
        embedding = self.model.encode(chunk_text)
        if len(embedding) != self.vector_size:
            raise ValueError(f"Expected embedding dimension {self.vector_size}, but got {len(embedding)}")
        embedding = embedding.tolist()

        # Payload
        point_id = str(uuid.uuid4())
        payload = {
            "document_id": document_id,
            "chunk_index": chunk_index,
            "text": node_id,
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
            print(f"Vector dimension: {len(embedding)}")
            raise

    def get_similar_chunks(self, query_text, document_id=None, limit=5):
        """Find chunks similar to the query text."""
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
            result_data = {
                "point_id": res.id,
                "similarity_score": res.score
            }
            result_data.update(res.payload)
            results.append(result_data)

        return results

    def get_document_chunks(self, document_id):
        """Retrieve all chunks for a specific document, ordered by chunk_index."""
        collection_name = "document_chunks"
        filter_param = models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        )
        scroll_results = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=filter_param,
            limit=10000
        )
        chunks = []
        for point in scroll_results[0]:
            chunk_data = {"point_id": point.id}
            chunk_data.update(point.payload)
            chunks.append(chunk_data)
        chunks.sort(key=lambda x: x["chunk_index"])
        return chunks

    def count_document_chunks(self, document_id):
        """Count the number of chunks in a document."""
        collection_name = "document_chunks"
        filter_param = models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        )
        count = self.client.count(collection_name=collection_name, count_filter=filter_param)
        return count.count


# ---------------------------------------------------------
# Neo4j ops
# ---------------------------------------------------------
def setup_neo4j_constraints():
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        try:
            session.run(
                "CREATE CONSTRAINT unique_sub_attr_name IF NOT EXISTS "
                "FOR (s:SubAttribute) "
                "REQUIRE (s.name, s.belongs_to_main) IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT unique_text_summary IF NOT EXISTS "
                "FOR (t:TextSummary) "
                "REQUIRE (t.content, t.describes_sub) IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT unique_main_attr_name IF NOT EXISTS "
                "FOR (m:MainAttribute) "
                "REQUIRE (m.name, m.part_of_doc) IS UNIQUE"
            )
        except Exception as e:
            print(f"Error setting up constraints: {e}")


def create_document_node(doc_name, properties=None):
    """Create a Document node to represent the PDF file."""
    node_id = str(uuid.uuid4())
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        cypher_query = "CREATE (d:Document {id: $id, name: $name}) RETURN d"
        params = {"id": node_id, "name": doc_name}

        if properties:
            property_set = ", ".join([f"d.{key} = ${key}" for key in properties.keys()])
            cypher_query = f"CREATE (d:Document {{id: $id, name: $name}}) SET {property_set} RETURN d"
            params.update(properties)

        session.run(cypher_query, params)

    return node_id


def create_main_attribute(name, document_id, properties=None):
    """Create a main attribute node connected to a document."""
    node_id = str(uuid.uuid4())
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        cypher_query = (
            "MATCH (d:Document {id: $doc_id}) "
            "CREATE (m:MainAttribute {id: $id, name: $name})-[:PART_OF]->(d) "
            "RETURN m"
        )
        params = {"id": node_id, "name": name, "doc_id": document_id}

        if properties:
            property_set = ", ".join([f"m.{key} = ${key}" for key in properties.keys()])
            cypher_query = (
                "MATCH (d:Document {id: $doc_id}) "
                "CREATE (m:MainAttribute {id: $id, name: $name})-[:PART_OF]->(d) "
                f"SET {property_set} "
                "RETURN m"
            )
            params.update(properties)

        session.run(cypher_query, params)

    return node_id


def create_sub_attribute(main_attr_id, name, properties=None):
    """Create a sub-attribute node connected to a main attribute."""
    node_id = str(uuid.uuid4())
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        cypher_query = (
            "MATCH (m:MainAttribute {id: $main_id}) "
            "CREATE (s:SubAttribute {id: $id, name: $name})-[:BELONGS_TO]->(m) "
            "RETURN s"
        )
        params = {"main_id": main_attr_id, "id": node_id, "name": name}

        if properties:
            property_set = ", ".join([f"s.{key} = ${key}" for key in properties.keys()])
            cypher_query = (
                "MATCH (m:MainAttribute {id: $main_id}) "
                "CREATE (s:SubAttribute {id: $id, name: $name})-[:BELONGS_TO]->(m) "
                f"SET {property_set} "
                "RETURN s"
            )
            params.update(properties)

        session.run(cypher_query, params)

    return node_id


def add_text_summary_with_merge(sub_attr_id, summary_name, text_content, properties=None):
    """MERGE a TextSummary (by content) and connect to SubAttribute."""
    node_id = str(uuid.uuid4())
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    try:
        uuid.UUID(sub_attr_id)
    except ValueError:
        print(f"Value of '{sub_attr_id}' is not a valid UUID:")
        return ""

    with driver.session() as session:
        cypher_query = (
            "MATCH (s:SubAttribute {id: $sub_id}) "
            "MERGE (t:TextSummary {content: $content})-[:DESCRIBES]->(s) "
            "ON CREATE SET t.id = $id, t.name = $name "
            "RETURN t.id as summary_id"
        )
        params = {"sub_id": sub_attr_id, "id": node_id, "name": summary_name, "content": text_content}

        if properties:
            property_set = ", ".join([f"t.{key} = ${key}" for key in properties.keys()])
            cypher_query = (
                "MATCH (s:SubAttribute {id: $sub_id}) "
                "MERGE (t:TextSummary {content: $content})-[:DESCRIBES]->(s) "
                "ON CREATE SET t.id = $id, t.name = $name "
                f"ON CREATE SET {property_set} "
                "RETURN t.id as summary_id"
            )
            params.update(properties)

        result = session.run(cypher_query, params)
        record = result.single()
        return record["summary_id"] if record else None


# ---------------------------------------------------------
# Insert data into Neo4j
# ---------------------------------------------------------
def insert_data(jsondata1, title, doc_id):
    description = ""
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        for category, terms in jsondata1.items():
            if category not in data_attributes:
                mainattr_id = create_main_attribute(category, doc_id, {"created_date": str(datetime.now())})
                data_main_attributes[category] = mainattr_id
                data_attributes[category] = {}
            if isinstance(terms, str):
                description = terms
                add_text_summary_with_merge(
                    data_main_attributes[category], title, description, {"created_date": str(datetime.now())}
                )
            else:
                for term, desc in terms.items():
                    if data_attributes[category] == {} or (term not in data_attributes[category]):
                        mainattr_id = data_main_attributes[category]
                        subattr_id = create_sub_attribute(mainattr_id, term, {"created_date": str(datetime.now())})
                        data_attributes[category][term] = subattr_id
                    add_text_summary_with_merge(
                        data_attributes[category][term], title, desc, {"created_date": str(datetime.now())}
                    )
    print("Nodes and relationships created successfully!")
    driver.close()


# ---------------------------------------------------------
# JSON validation + response parsing helpers
# ---------------------------------------------------------
def validate_json_structure(json_str_val):
    try:
        data = json.loads(json_str_val)

        if len(data) < 1:
            return True, "JSON should have at least one top-level attribute"

        for main_attr_name, main_attr in data.items():
            if not isinstance(main_attr, dict):
                return False, f"The main attribute '{main_attr_name}' should be an object"

            for sub_attr, value in main_attr.items():
                if not isinstance(value, str):
                    return False, f"Value of '{sub_attr}' in '{main_attr_name}' is not a string"

        return True, "JSON structure is valid"

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def process_response(response_1):
    tag = "</think>"
    text = ""
    tag_position = response_1.find(tag)
    after = ""

    if tag_position != -1:
        text = response_1[tag_position + len(tag):]
    else:
        text = response_1

    if text.find("{"):
        before, after = text.split("{", 1)
    after = "{" + after

    after_rr = after.rstrip('`').strip()

    valid_response_1, validation_reason = validate_json_structure(after_rr)
    repeat_boolean_1 = not valid_response_1

    return after_rr, repeat_boolean_1, valid_response_1


# ---------------------------------------------------------
# Shared counters
# ---------------------------------------------------------
counter_lock = threading.Lock()
COUNT = 0


def increment_counter():
    global COUNT
    with counter_lock:
        current = COUNT
        COUNT += 1
        return current


# ---------------------------------------------------------
# Neo4j retrieval for summaries
# ---------------------------------------------------------
def get_summaries_by_subcategory_id(subcategory_id):
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    with driver.session() as session:
        query = """
        MATCH (s:SubAttribute {id: $subcategory_id})
        OPTIONAL MATCH (s)<-[:DESCRIBES]-(t:TextSummary)
        WITH s, collect(t) as summaries
        RETURN s.id as subcategory_id, 
               s.name as subcategory_name, 
               s.priority as priority,
               summaries
        """
        result = session.run(query, subcategory_id=subcategory_id)
        record = result.single()
        if not record:
            return None

        word_count = 0
        summary_list = ""

        print(f"SubAttribute: {record['subcategory_name']}")
        for summary in record["summaries"]:
            if summary is not None:
                if "content" in summary:
                    content = summary["content"]
                    summary_list += content + "\n"
                    word_count += len(content.split())

        print("word count: " + str(word_count))
        return [word_count, summary_list]


# ---------------------------------------------------------
# LLM processing per page
# ---------------------------------------------------------
def process_text_from_pdf(text, page_num, doc_id):
    valid_response = False
    after_process_text = ""
    repeat_boolean = True
    global response

    while repeat_boolean:
        your_question = (
            "Given the following list of main attributes and their sub-attributes:\n"
            f"{json.dumps(data_attributes)}\n\n"
            "Your task is to analyze the entire text below and comprehensively map it to the most relevant main attributes and sub-attributes from the list.\n\n"
            "Instructions:\n"
            "1. ONLY select from the provided attribute list when a relevant match is found in the text.\n"
            "2. If no suitable match exists, create a new main attribute and clear, meaningful sub-attributes — avoid placeholders like 'NEW_ATTRIBUTE'.\n"
            "3. You MUST identify and return all relevant sub-attributes — do not stop at the first match.\n"
            "4. For each sub-attribute, extract a short but specific description or excerpt from the original text that directly supports the match.\n"
            "   - Do NOT summarize vaguely. Include all essential facts, clauses, or phrases that explain the sub-attribute's relevance.\n"
            "   - If multiple relevant details exist in the text, merge them logically into a single description.\n"
            "   - Do not miss or skip over any informative parts tied to the sub-attribute.\n"
            "5. Ignore content that does not meaningfully relate to the attribute list.\n"
            "6. Return ONLY a valid JSON object in this format — no explanation, no additional text:\n"
            "{\n"
            "  \"<main_attribute>\": {\n"
            "    \"<sub_attribute_1>\": \"<relevant description from text>\",\n"
            "    \"<sub_attribute_2>\": \"<relevant description from text>\"\n"
            "  },\n"
            "  ... (more attributes)\n"
            "}\n\n"
            f"TEXT TO ANALYZE:\n{text}\n\n"
            "IMPORTANT:\n"
            "- Be exhaustive: extract all meaningful matches across the entire text.\n"
            "- Relevance threshold for matching: above 70% semantic similarity.\n"
            "- For any unmatched but meaningful concept, create a new attribute and sub-attributes.\n"
            "- The result must be clean, structured, and strictly JSON formatted — no commentary.\n"
            "- If nothing matches, return an empty JSON object: {}"
        )

        payload = {"contents": [{"parts": [{"text": your_question}]}]}
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                GEMINI_URL + GEMINI_API_KEY[increment_counter() % g_l],
                headers=headers,
                data=json.dumps(payload)
            )
            raw_text_1 = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            after_process_text, repeat_boolean, valid_response = process_response(raw_text_1)
        except Exception as e:
            print("found exception: ", e)
            valid_response = False

        if valid_response:
            insert_data(json.loads(after_process_text.replace("\n", "")), "Description_" + str(page_num), doc_id)
        else:
            print("INVALID json from llm ===========================================================================")
            print(response)
            print("count:  " + str(COUNT))

    return json.loads(after_process_text.replace("\n", ""))


# ---------------------------------------------------------
# PDF extraction + embeddings
# ---------------------------------------------------------
def extract_text_from_pdf(pdf_path, doc_name):
    document_id = create_document_node(doc_name, {"created_date": str(datetime.now())})
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        print(len(reader.pages))
        futures = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                future = executor.submit(process_text_from_pdf, text, page_num, document_id)
                futures.append(future)
            for future in as_completed(futures):
                r = future.result()
                print(r)

        with open(f"{doc_name}.json", "w") as file_1:
            json.dump(data_attributes, file_1)
        with open(f"main_attribute_{doc_name}.json", "w") as file_2:
            json.dump(data_main_attributes, file_2)

    return "Extraction completed"


def execute_embeddings_from_text(system, node_id, doc_id, category, sub_category):
    summaries = get_summaries_by_subcategory_id(node_id)
    final_summery = (
        f"Category: {category}"
        f"Subcategory: {sub_category}"
        f"Summary: {summaries[1]}"
    )

    if not summaries or len(summaries) < 2 or not summaries[1]:
        return f"No valid text content found for node {node_id}"

    try:
        chunk_id = system.add_chunk(
            document_id=doc_id,
            chunk_text=final_summery,
            node_id=node_id
        )
        return f"Added chunk {chunk_id} to document {doc_id} for node {node_id}"
    except Exception as e:
        return f"Error adding chunk for node {node_id}: {str(e)}"


def parse_embeddings_from_text(doc_id):
    stats = {
        "total_nodes": 0,
        "successful": 0,
        "failed": 0,
        "errors": []
    }

    try:
        system = DocumentEmbeddingSystem()
    except Exception as e:
        print(f"Failed to initialize DocumentEmbeddingSystem: {str(e)}")
        return {"error": f"Failed to initialize embedding system: {str(e)}"}

    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_EMBEDDINGS) as executor:
        for main, sub_attrs in data_attributes.items():
            for sub_attr, node_id in sub_attrs.items():
                stats["total_nodes"] += 1
                future = executor.submit(
                    execute_embeddings_from_text, system, node_id, doc_id, main, sub_attr
                )
                futures.append((future, node_id))

        for future, node_id in [(f, nid) for f, nid in futures]:
            try:
                result = future.result()
                print(result)
                if "Error" in result:
                    stats["failed"] += 1
                    stats["errors"].append({"node_id": node_id, "error": result})
                else:
                    stats["successful"] += 1
            except Exception as e:
                stats["failed"] += 1
                error_msg = f"Exception processing node {node_id}: {str(e)}"
                print(error_msg)
                stats["errors"].append({"node_id": node_id, "error": error_msg})

    return stats


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------
def run_pipeline(pdf_path, doc_name):
    start_time = time.time()
    extract_text_from_pdf(pdf_path, doc_name)
    print("Extraction done. Time taken:", time.time() - start_time)

    parse_embeddings_from_text(doc_name)
    print("Full pipeline completed. Total time:", time.time() - start_time)
    print("time taken: " + str(time.time() - start_time))


def extract(pdf_path: str, doc_name: str) -> str:
    return run_pipeline(pdf_path, doc_name)


# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
extract("/Users/pranaligole/MS/Courses/Sem2/CS532/Code/Data/EOC_2.pdf", "B")
