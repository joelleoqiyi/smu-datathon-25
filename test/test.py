import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

# Load the pre-trained NER model from Hugging Face
model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a Named Entity Recognition pipeline
nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Load the dataset
file_path = "news_excerpts_parsed.xlsx"  # Update with your actual file path
df = pd.read_excel(file_path)

# Function to perform NER on text
def extract_ner_entities(text):
    if isinstance(text, str):  # Ensure the input is a string
        ner_results = nlp_pipeline(text)
        extracted_entities = [(entity['word'], entity['entity'], entity['score']) for entity in ner_results]
        return extracted_entities
    return []

# Apply NER extraction to each text entry
df["NER_Entities"] = df["Text"].apply(extract_ner_entities)

# Save results to a new Excel file
df.to_excel("ner_extracted_results.xlsx", index=False)

print("NER extraction complete. Results saved to 'ner_extracted_results.xlsx'.")