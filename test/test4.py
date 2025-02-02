import pandas as pd
import numpy as np
from collections import defaultdict
from fuzzywuzzy import process, fuzz
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering

# Load the dataset
file_path = "cleaned_ner_results.xlsx"  # Update this with the actual file path
df = pd.read_excel(file_path)

# ✅ 1. Load the sentence transformer model (Fast, 6x speedup)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ 2. Extract all unique entity names dynamically from the dataset
all_entities = set()
for entity_list in df["Final_NER_Entities"]:  # Change this if your column name is different
    for name, entity_type, _ in eval(entity_list):  # Convert string to list
        all_entities.add(name)

# ✅ 3. Compute embeddings for all entity names at once (100x faster than loops)
entity_list = list(all_entities)
entity_embeddings = embedding_model.encode(entity_list, convert_to_tensor=True)

# ✅ 4. Use clustering to pre-group similar entities (Agglomerative Clustering)
def cluster_entities(entities, embeddings, threshold=0.85):
    """
    - Uses Agglomerative Clustering to group similar entities
    - Reduces the number of comparisons needed
    """
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - threshold, linkage='average', metric='cosine')
    labels = clustering.fit_predict(embeddings.cpu().numpy())

    # Create a mapping from entity to its cluster
    cluster_map = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_map[label].append(entities[idx])

    # Pick the most descriptive name from each cluster
    alias_map = {}
    for cluster in cluster_map.values():
        canonical_name = max(cluster, key=len)  # Choose the longest name as canonical
        for entity in cluster:
            alias_map[entity] = canonical_name

    return alias_map

# ✅ 5. Generate the alias mapping using clustering (FAST, scalable)
entity_aliases = cluster_entities(entity_list, entity_embeddings)

# ✅ 6. Function to normalize entity names using the alias mapping
def normalize_entity(entity_name):
    return entity_aliases.get(entity_name, entity_name)  # Replace if alias exists

# ✅ 7. Apply alias mapping to NER entities
df["Normalized_NER_Entities"] = df["Final_NER_Entities"].apply(
    lambda entities: [
        (normalize_entity(name), entity_type, confidence)
        for name, entity_type, confidence in eval(entities)
    ]
)

# ✅ 8. Save standardized entity results to a new file
df.to_excel("optimized_normalized_ner.xlsx", index=False)
print("✅ Auto-aliasing complete! Results saved to 'optimized_normalized_ner.xlsx'.")