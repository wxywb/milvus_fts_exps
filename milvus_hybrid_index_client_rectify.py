import random
import string
import numpy as np
import json

from pymilvus import MilvusClient, DataType, Function, FunctionType, utility

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.dense import  VoyageEmbeddingFunction

from typing import List, Dict, Any, Callable


storage = []

class HybridRetriever:
    def __init__(self,
                uri,
                collection_name = "hybrid",
                dense_embedding_function=None):
        self.uri = uri
        self.collection_name = collection_name
        self.embedding_function = dense_embedding_function
        self.use_reranker = True
        self.use_sparse = True
        self.client = MilvusClient(uri=uri)

    def build_collection(self):
        if isinstance(self.embedding_function.dim, dict):
            dense_dim = self.embedding_function.dim["dense"]
        else:
            dense_dim = self.embedding_function.dim
        
        analyzer_params_built_in = {
            "type": "english"
        }

        tokenizer_params = {
                "tokenizer": "standard",
                "filter":["lowercase", 
                    {
                        "type": "length",
                        "max": 200,
                    },{
                        "type": "stemmer",
                        "language": "english"
                    },{
                        "type": "stop",
                        "stop_words": [
                            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", 
                            "no", "not", "of","how","what","where", "does","can", "do",  "on", "or", "such", "that", "the", "their", "then", "there", "these", 
                            "they", "this", "to", "was", "will", "with","I", "get"
                        ],
                    }],
            }
            
        schema = MilvusClient.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="content",datatype=DataType.VARCHAR, max_length=65535, analyzer_params=tokenizer_params, enable_match=True,  enable_analyzer=True)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field(field_name="original_uuid", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=64),
        schema.add_field(field_name="original_index", datatype=DataType.INT32)


        functions = Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names="sparse_vector",
        )
        
        schema.add_function(functions)

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        index_params.add_index(field_name="dense_vector", index_type="FLAT", metric_type="IP")

        self.client.create_collection(collection_name=self.collection_name, schema=schema, index_params=index_params)

    def insert_data(self, chunk, metadata):
        embedding = self.embedding_function([chunk])
        print(embedding)
        if isinstance(embedding, dict) and 'dense' in embedding:
            dense_vec = embedding['dense'][0] 
        else:
            dense_vec = embedding[0]
        self.client.insert(self.collection_name, {"dense_vector": dense_vec, **metadata})

 
if __name__ == '__main__':

    dense_ef = VoyageEmbeddingFunction(api_key=os.getenv("VOYAGE_API"), model_name="voyage-2")
    standard_retriever = HybridRetriever(
        uri="http://localhost:19530", collection_name="milvus_rectify", dense_embedding_function=dense_ef
    )

    path = "codebase_chunks.json"
    with open(path, "r") as f:
        dataset = json.load(f)

    is_insert = True
    if is_insert:
        standard_retriever.build_collection() 
        for doc in dataset:
            doc_content = doc["content"]
            for chunk in doc["chunks"]:
                metadata = {
                    "doc_id": doc["doc_id"],
                    "original_uuid": doc["original_uuid"],
                    "chunk_id": chunk["chunk_id"],
                    "original_index": chunk["original_index"],
                    "content": chunk["content"],
                }
                chunk_content = chunk["content"]
                standard_retriever.insert_data(chunk_content, metadata)


