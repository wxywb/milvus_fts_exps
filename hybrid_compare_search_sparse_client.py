import pickle
import json
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pymilvus import (
    MilvusClient,
    AnnSearchRequest,
    RRFRanker,
)
from pymilvus.model.dense import  VoyageEmbeddingFunction

def load_jsonl(file_path: str):
    """Load JSONL file and return a list of dictionaries."""
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


if __name__ == '__main__':
    uri="http://localhost:19530"
    good_count = 0
    bad_count = 0

    collection_name="milvus_standard"

    fw_good = open('sparse_good_case.txt', 'w')
    fw_bad = open('sparse_bad_case.txt', 'w')

    dense_ef = VoyageEmbeddingFunction(api_key=os.getenv("VOYAGE_API"), model_name="voyage-2")
    client = MilvusClient(uri=uri)

    output_fields=[
            "content",
            "original_uuid",
            "doc_id",
            "chunk_id",
            "original_index",
    ]

    dataset = load_jsonl("evaluation_set.jsonl")

    k = 5

    total_hybrid_query_score = 0
    total_sparse_query_score = 0
    num_queries = 0
    for query_item in dataset:
        query = query_item['query']
            
        golden_chunk_uuids = query_item['golden_chunk_uuids']
        
        golden_contents = []
        for doc_uuid, chunk_index in golden_chunk_uuids:
            golden_doc = next((doc for doc in query_item['golden_documents'] if doc['uuid'] == doc_uuid), None)
            if golden_doc:
                golden_chunk = next((chunk for chunk in golden_doc['chunks'] if chunk['index'] == chunk_index), None)
                if golden_chunk:
                    golden_contents.append(golden_chunk['content'].strip())

        embedding = dense_ef([query])

        if isinstance(embedding, dict) and  'dense' in embedding:
            dense_vec = embedding['dense'][0] 
        else:
            dense_vec = embedding[0]

        full_text_search_params = {"metric_type": "BM25"}
        full_text_search_req = AnnSearchRequest([query], "sparse_vector", full_text_search_params, limit=k)
        dense_search_params = {"metric_type": "IP"}
        dense_req = AnnSearchRequest(
            [dense_vec], "dense_vector", dense_search_params, limit=k
        )

        hybrid_docs = client.hybrid_search(
            collection_name, [full_text_search_req, dense_req], ranker=RRFRanker(), limit=k, output_fields=output_fields
        )
        hybrid_results = [{'doc_id': doc["entity"]["doc_id"], 'chunk_id': doc["entity"]["chunk_id"], 'content': doc["entity"]["content"], 'score':doc["distance"]} for doc in hybrid_docs[0]]
    
        #sparse
        sparse_docs = client.search(collection_name, data=[query], anns_field="sparse_vector", limit=k, output_fields=output_fields)  
        sparse_results = [{'doc_id': doc["entity"]["doc_id"], 'chunk_id': doc["entity"]["chunk_id"], 'content': doc["entity"]["content"], 'score':doc["distance"]} for doc in sparse_docs[0]]

        sparse_chunks_found = 0
        hybrid_chunks_found = 0


        gts = []
        sparse_gts = []
        for golden_content in golden_contents: 
            for doc in sparse_results[:k]: 
                retrieved_content = doc['content'].strip()
                if retrieved_content == golden_content:
                    sparse_gts.append(retrieved_content)
                    sparse_chunks_found += 1
                    break
        for golden_content in golden_contents:
            for doc in hybrid_results[:k]:
                retrieved_content = doc['content'].strip()
                if retrieved_content == golden_content:
                    gts.append(retrieved_content)
                    hybrid_chunks_found += 1
                    break

        sparse_query_score = sparse_chunks_found / len(golden_contents)
        hybrid_query_score = hybrid_chunks_found / len(golden_contents)

        total_sparse_query_score += sparse_query_score
        total_hybrid_query_score += hybrid_query_score
        num_queries += 1
        print(num_queries, query)
        print(num_queries, 'FTS Pass@5:', total_sparse_query_score/num_queries, 'hybrid Pass@5:', total_hybrid_query_score/num_queries, sparse_chunks_found, hybrid_chunks_found)
        

        if hybrid_chunks_found > sparse_chunks_found:
            fw_good.write(f'{good_count} {query} \n')
            fw_good.write('--------------------------\n')
            for gti, gt in enumerate(gts):
                fw_good.write(f'gt {gti} \n')
                fw_good.write(f"{gt}\n")

            for i, sparse_result in enumerate(sparse_results):
                fw_good.write(f'##sparse {i} {sparse_result["score"]} \n')
                fw_good.write(f"{sparse_result['content']}\n")
                fw_good.write("\n ")

            for i, hybrid_result in enumerate(hybrid_results):
                fw_good.write(f'##hybrid {i} {hybrid_result["score"]} \n')
                fw_good.write(f"{hybrid_result['content']}\n")
                fw_good.write(" \n")

            good_count = good_count + 1

        if hybrid_chunks_found < sparse_chunks_found:
            fw_bad.write(f'{bad_count} {query} \n')
            fw_bad.write('-------------------------- \n')
            for gti, gt in enumerate(sparse_gts):
                fw_bad.write(f'gt {gti} \n')
                fw_bad.write(f"{gt}\n")
        
            for i, sparse_result in enumerate(sparse_results):
                fw_bad.write(f'##sparse {i} {sparse_result["score"]} \n')
                fw_bad.write(f"f{sparse_result['content']}\n")
                fw_bad.write("\n ")
        
            for i, hybrid_result in enumerate(hybrid_results):
                fw_bad.write(f'##hybrid {i} {hybrid_result["score"]} \n')
                fw_bad.write(f"{hybrid_result['content']}\n")
                fw_bad.write("\n ")
        
            bad_count = bad_count + 1


