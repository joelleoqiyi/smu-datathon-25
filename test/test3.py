# Import required libraries
import pandas as pd
import numpy as np
from collections import defaultdict
from fuzzywuzzy import process, fuzz
from sentence_transformers import SentenceTransformer, util

# Load the dataset
file_path = "cleaned_ner_results.xlsx"  # Update this with the actual file path
df = pd.read_excel(file_path)

print("✅ 1. Load a sentence transformer model for semantic similarity")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ 2. Extract all unique entity names dynamically from the dataset")
all_entities = set()
for entity_list in df["Final_NER_Entities"]:
    for name, entity_type, _ in eval(entity_list):  # Convert string to list
        all_entities.add(name)

print("✅ 3. Compute word embeddings for all entity names using SentenceTransformer")
entity_embeddings = {
    entity: embedding_model.encode(entity, convert_to_tensor=True)
    for entity in all_entities
}

print("✅ 4. Automatically generate alias mappings using fuzzy matching and semantic similarity")
def build_alias_map(entities, threshold_fuzzy=85, threshold_semantic=0.85):
    """
    - Uses fuzzy matching to find similar text-based entities.
    - Uses cosine similarity to find semantically similar entities.
    - Groups variations under a single canonical entity name.
    """
    alias_map = {}
    processed = set()
    entity_list = list(entities)
    print("Number of entities to complete: ", len(entity_list))

    for i, entity in enumerate(entity_list):
        print("Completed entity: ", i)
        if entity in processed:
            continue

        # ✅ 4.1 Fuzzy matching to find textually similar entities
        match, score = process.extractOne(entity, entity_list[i+1:]) if entity_list[i+1:] else (None, 0)

        # ✅ 4.2 Compute semantic similarity (cosine similarity of embeddings)
        best_semantic_match = None
        best_semantic_score = 0

        for other_entity in entity_list:
            if entity == other_entity:
                continue
            semantic_score = util.pytorch_cos_sim(
                entity_embeddings[entity], entity_embeddings[other_entity]
            ).item()
            if semantic_score > best_semantic_score:
                best_semantic_match = other_entity
                best_semantic_score = semantic_score

        # ✅ 4.3 Choose the best match (either fuzzy or semantic)
        best_match = None
        if match and score >= threshold_fuzzy:
            best_match = match
        if best_semantic_match and best_semantic_score >= threshold_semantic:
            best_match = best_semantic_match

        # ✅ 4.4 If a valid match is found, group them under the longer/more formal name
        if best_match:
            canonical_name = max(entity, best_match, key=len)  # Keep the longer/more descriptive name
            alias_map[entity] = canonical_name
            alias_map[best_match] = canonical_name
            processed.add(entity)
            processed.add(best_match)

    return alias_map

print("✅ 5. Generate the alias mapping dynamically")
entity_aliases = build_alias_map(all_entities)

print("✅ 6. Function to apply alias mapping")
def normalize_entity(entity_name):
    return entity_aliases.get(entity_name, entity_name)  # Replace if alias exists

print("✅ 7. Apply alias mapping to NER entities")
df["Normalized_NER_Entities"] = df["Final_NER_Entities"].apply(
    lambda entities: [
        (normalize_entity(name), entity_type, confidence)
        for name, entity_type, confidence in eval(entities)
    ]
)

print("✅ 8. Save standardized entity results to a new file")
df.to_excel("generalized_normalized_ner_with_embeddings.xlsx", index=False)
print("✅ Auto-aliasing with word embeddings & fuzzy matching complete!")

# -------------------------------------------------------------
# ✅ RELATIONSHIP EXTRACTION (AFTER ENTITY STANDARDIZATION)
# -------------------------------------------------------------

# ✅ 9. Function to extract relationships while ensuring standardization
def extract_relationships(text, entities):
    """
    - Extracts relationships between entities within the same excerpt.
    - Uses standardized entity names.
    """
    relationships = []
    entity_names = [name for name, _, _ in entities]  # Extract only entity names
    
    # ✅ 9.1 Generate simple entity-entity relationships (for each entity pair)
    for i in range(len(entity_names)):
        for j in range(i + 1, len(entity_names)):
            relationships.append((entity_names[i], entity_names[j]))  # (Entity1, Entity2)
    
    return relationships

# ✅ 10. Apply relationship extraction using normalized entity names
df["Relationships"] = df.apply(
    lambda row: extract_relationships(row["Text"], row["Normalized_NER_Entities"]),
    axis=1
)

# ✅ 11. Aggregate relationships across all excerpts
relationship_dict = defaultdict(set)

for relationships in df["Relationships"]:
    for entity1, entity2 in relationships:
        relationship_dict[entity1].add(entity2)
        relationship_dict[entity2].add(entity1)  # Bidirectional relationship

# ✅ 12. Convert relationships into a DataFrame for better visualization
relationship_df = pd.DataFrame(
    [(k, list(v)) for k, v in relationship_dict.items()], 
    columns=["Entity", "Related Entities"]
)

# ✅ 13. Save relationships to a new Excel file
relationship_df.to_excel("aggregated_relationships.xlsx", index=False)
print("✅ Relationship extraction complete! Results saved to 'aggregated_relationships.xlsx'.")