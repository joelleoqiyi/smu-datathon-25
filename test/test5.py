# Import required libraries
import pandas as pd
import numpy as np
from collections import defaultdict
from fuzzywuzzy import process, fuzz
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Load the dataset
file_path = "news_excerpts_parsed.xlsx"  # Update this with the actual file path
df = pd.read_excel(file_path)

# ########################################################
# ✅ 1. Named Entity Recognition (NER) Extraction
# ########################################################

# Load the pre-trained NER model
model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a Named Entity Recognition pipeline
nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# ✅ Function to perform NER extraction & merge subwords (### tokens)
def extract_ner_entities(text):
    if isinstance(text, str):
        ner_results = nlp_pipeline(text)
        
        merged_entities = []
        current_entity = []
        current_label = None

        for entity in ner_results:
            word = entity['word']
            entity_type = entity['entity_group']
            confidence = round(entity['score'], 4)

            # Handle subword tokens (e.g., 'U.', '##OB' -> "UOB")
            if word.startswith("##"):
                if current_entity:
                    current_entity[-1] += word[2:]  # Merge subword into previous token
            else:
                if current_entity:
                    merged_entities.append(("".join(current_entity), current_label, max_confidence))
                current_entity = [word]
                current_label = entity_type
                max_confidence = confidence

        # Add the last entity if present
        if current_entity:
            merged_entities.append(("".join(current_entity), current_label, max_confidence))

        return merged_entities
    return []

# Apply NER extraction to each text entry
df["NER_Entities"] = df["Text"].apply(extract_ner_entities)

# ########################################################
# ✅ 2. Entity Standardization (Fuzzy Matching + Clustering)
# ########################################################

# ✅ Load the sentence transformer model for entity similarity
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Extract all unique entity names dynamically from the dataset
all_entities = set()
for entity_list in df["NER_Entities"]:
    for name, entity_type, _ in entity_list:
        all_entities.add(name)

# ✅ Compute embeddings for all entity names (Batch processing)
entity_list = list(all_entities)
entity_embeddings = embedding_model.encode(entity_list, convert_to_tensor=True)

# ✅ Function to cluster entities and standardize names
def cluster_entities(entities, embeddings, threshold=0.85):
    """
    - Uses Agglomerative Clustering to group similar entities
    - Reduces the number of comparisons needed
    """
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1 - threshold, linkage='average', metric='cosine'
    )
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

# ✅ Generate the alias mapping dynamically
entity_aliases = cluster_entities(entity_list, entity_embeddings)

# ✅ Function to normalize entity names using alias mapping
def normalize_entity(entity_name):
    return entity_aliases.get(entity_name, entity_name)

# ✅ Apply alias mapping to NER entities
df["Normalized_NER_Entities"] = df["NER_Entities"].apply(
    lambda entities: [
        (normalize_entity(name), entity_type, confidence)
        for name, entity_type, confidence in entities
    ]
)

# ✅ Save standardized entity results to a new file
df.to_excel("optimized_normalized_ner.xlsx", index=False)
print("✅ Auto-aliasing complete! Results saved to 'optimized_normalized_ner.xlsx'.")

# ########################################################
# ✅ 3. Relationship Extraction (After Entity Standardization)
# ########################################################

# ✅ Function to extract relationships while ensuring standardization
def extract_relationships(text, entities):
    """
    - Extracts relationships between entities within the same excerpt.
    - Uses standardized entity names.
    """
    relationships = []
    entity_names = [name for name, _, _ in entities]  # Extract only entity names
    
    # ✅ Generate simple entity-entity relationships (for each entity pair)
    for i in range(len(entity_names)):
        for j in range(i + 1, len(entity_names)):
            relationships.append((entity_names[i], entity_names[j]))  # (Entity1, Entity2)
    
    return relationships

# ✅ Apply relationship extraction using normalized entity names
df["Relationships"] = df.apply(
    lambda row: extract_relationships(row["Text"], row["Normalized_NER_Entities"]),
    axis=1
)

# ✅ Aggregate relationships across all excerpts
relationship_dict = defaultdict(set)

for relationships in df["Relationships"]:
    for entity1, entity2 in relationships:
        relationship_dict[entity1].add(entity2)
        relationship_dict[entity2].add(entity1)  # Bidirectional relationship

# ✅ Convert relationships into a DataFrame for better visualization
relationship_df = pd.DataFrame(
    [(k, list(v)) for k, v in relationship_dict.items()], 
    columns=["Entity", "Related Entities"]
)

# ✅ Save relationships to a new Excel file
relationship_df.to_excel("aggregated_relationships.xlsx", index=False)
print("✅ Relationship extraction complete! Results saved to 'aggregated_relationships.xlsx'.")