import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Load the pre-trained NER model from Hugging Face
model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a Named Entity Recognition pipeline (with auto-aggregation)
nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Load the dataset
file_path = "news_excerpts_parsed.xlsx"  # Update with actual path
df = pd.read_excel(file_path)

# Function to extract unique NER entities with the highest confidence score
def extract_unique_ner_entities(text):
    if isinstance(text, str):  # Ensure input is a valid string
        ner_results = nlp_pipeline(text)
        
        entity_dict = {}  # Dictionary to track the highest confidence score per entity
        for entity in ner_results:
            entity_name = entity['word']
            entity_type = entity['entity_group']
            confidence = round(entity['score'], 4)
            
            # Store only the highest confidence occurrence
            if entity_name not in entity_dict or confidence > entity_dict[entity_name][1]:
                entity_dict[entity_name] = (entity_type, confidence)

        # Convert back to list format: [(Entity, Type, Confidence)]
        return [(name, details[0], details[1]) for name, details in entity_dict.items()]
    
    return []

# Apply the function to extract and filter unique NER entities
df["Final_NER_Entities"] = df["Text"].apply(extract_unique_ner_entities)

# Save results to a new Excel file
df.to_excel("cleaned_ner_results.xlsx", index=False)

print("✅ NER extraction complete. Duplicates removed. Results saved to 'cleaned_ner_results.xlsx'.")