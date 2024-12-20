# Milvus Full Text Search and Hybrid Search Experiments

This repository demonstrates experiments using Milvus for **Full Text Search** (FTS) and **Hybrid Search** capabilities. We explore the comparison between **dense search**, **sparse search**, and **hybrid search** for retrieving relevant data.

## Prerequisite
Deploy [Milvus 2.5](https://milvus.io/docs/install_standalone-docker-compose.md) 

Install PyMivus with Model Library.
```bash
pip install "pymilvus[model]" -U
```

Get a [Voyage AI](https://www.voyageai.com/) API Key.


## Setup

Before running any experiments, ensure that the required environment variables and collections are properly set up.

### 1. Set up `VOYAGE_API` environment variable

You need to set up your `VOYAGE_API` as an environment variable for authentication.

```bash
export VOYAGE_API="your_api_key_here"
```

### 2. Insert Data into Milvus
The experiments require data to be inserted into Milvus. The following script will insert data into the collection `milvus_standard`:

```bash
python milvus_hybrid_index_client.py
```
### 3. Hybrid vs Dense Search Comparison
To compare Hybrid Search and Dense Search, run the following script:

```bash
python hybrid_compare_search_dense_client.py
```
This will output results comparing the performance and retrieval quality of hybrid search against dense vector search.

### 4. Hybrid vs Sparse Search Comparison
Similarly, to compare Hybrid Search and Sparse Search, run the following script:

```bash
python hybrid_compare_search_sparse_client.py
```
### 5. Insert Data with Modified Stopwords
To insert data into Milvus with modified stopwords, use the script below. This will insert data into the collection milvus_rectify:

```bash
python milvus_hybrid_index_client_rectify.py
```
### 6. Modify Collection Name for Comparison
After inserting data with modified stopwords, you need to modify the collection name to milvus_rectify in the following scripts:

`hybrid_compare_search_dense_client.py`
`hybrid_compare_search_sparse_client.py`
Manually change the collection name in the script from `milvus_standard` to `milvus_rectify`.

### 7. Re-run the Comparison Scripts
After updating the collection name, execute the following scripts again to observe the effect of modified stopwords:

Hybrid vs Dense Comparison:


```bash
python hybrid_compare_search_dense_client.py
```
Hybrid vs Sparse Comparison:

```bash
python hybrid_compare_search_sparse_client.py
```
### 8. Results and Observations
After running the comparisons, analyze the results to see how the different search types (Hybrid, Dense, Sparse) perform, especially when modified stopwords are introduced.





