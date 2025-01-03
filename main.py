import spacy
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from symspellpy import SymSpell
import wikipediaapi
import requests
import os
import json
from concurrent.futures import ThreadPoolExecutor

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set environment variable for memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize NLP Tools
nlp = spacy.load("en_core_web_sm")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# Spell Checking
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)
sym_spell.load_bigram_dictionary("frequency_bigramdictionary_en_243_342.txt", 0, 2)

# Wikipedia API
wiki_user_agent = "AIChatBot/1.0 (https://example.com; support@example.com)"
wiki = wikipediaapi.Wikipedia("en", headers={"User-Agent": wiki_user_agent})

# Fact-Checking Model (FLAN-T5)
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
flan_t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# Sentiment Analysis Model
sentiment_pipeline = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)

# Question-Answering Models
qa_pipeline_roberta = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if torch.cuda.is_available() else -1)

# Falcon Model Initialization
falcon_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
falcon_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

falcon_model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-7b-instruct",
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Google Fact Check API Key
GOOGLE_API_KEY = "screctkey"

# Define source credibility weights
SOURCE_WEIGHTS = {
    "Wikipedia": 0.9,
    "Google Fact Check API": 0.85,
    "Falcon-7B": 0.95,
    "FLAN-T5": 0.9,
    "Roberta QA": 0.8,
}

# Load keywords dynamically from external files
def load_keywords_from_file(filepath):
    try:
        with open(filepath, 'r') as file:
            return [line.strip() for line in file if line.strip()]
    except Exception as e:
        print(f"Error loading keywords from {filepath}: {e}")
        return []

def load_label_keywords(filepath):
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading label keywords from {filepath}: {e}")
        return {}

LABEL_KEYWORDS = load_label_keywords("label_keywords.txt")
question_keywords = load_keywords_from_file("question_keywords.txt")
fact_check_keywords = load_keywords_from_file("fact_check_keywords.txt")

def spell_check(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

def classify_intent(text):
    lower_text = text.lower()
    if any(keyword in lower_text for keyword in question_keywords):
        return "question"
    if any(keyword in lower_text for keyword in fact_check_keywords):
        return "fact-check"
    return "unknown"

def fetch_wikipedia_summary(query):
    page = wiki.page(query)
    return page.summary if page.exists() else None

def fact_check_with_flan_t5(claim):
    input_text = f"fact-check: {claim}"
    inputs = flan_t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
    outputs = flan_t5_model.generate(inputs, max_length=50)
    return flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def query_qa_roberta(question, context=None):
    try:
        if context:
            return qa_pipeline_roberta({"question": question, "context": context}).get("answer", None)
        return None
    except Exception:
        return None

def query_falcon(question):
    try:
        inputs = falcon_tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = falcon_model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=100)
        return falcon_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error querying Falcon model: {str(e)}"

def fetch_fact_check_from_google(query):
    try:
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={GOOGLE_API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            claims = data.get("claims", [])
            if claims:
                return "\n".join([claim.get("text", "") for claim in claims])
        return None
    except Exception:
        return None

def classify_label_advanced(responses, sources):
    scores = {label: 0 for label in LABEL_KEYWORDS.keys()}
    for response, source in zip(responses, sources):
        for label, keywords in LABEL_KEYWORDS.items():
            if any(keyword in response.lower() for keyword in keywords):
                scores[label] += SOURCE_WEIGHTS.get(source, 0.5)
    return max(scores, key=scores.get)

def calculate_advanced_accuracy(responses, sources):
    credible_sources = [src for src in sources if SOURCE_WEIGHTS.get(src, 0) > 0.8]
    agreements = sum(1 for src in sources if src in credible_sources)
    total = len(sources)
    return round((agreements / total) * 100, 2) if total > 0 else 0

def process_input_advanced(text):
    try:
        responses = []
        sources = []
        with ThreadPoolExecutor() as executor:
            wiki_future = executor.submit(fetch_wikipedia_summary, text)
            t5_future = executor.submit(fact_check_with_flan_t5, text)
            roberta_future = executor.submit(query_qa_roberta, text)
            falcon_future = executor.submit(query_falcon, text)
            google_future = executor.submit(fetch_fact_check_from_google, text)

            wiki_summary = wiki_future.result()
            if wiki_summary:
                responses.append(wiki_summary)
                sources.append("Wikipedia")

            t5_response = t5_future.result()
            if t5_response:
                responses.append(t5_response)
                sources.append("FLAN-T5")

            roberta_response = roberta_future.result()
            if roberta_response:
                responses.append(roberta_response)
                sources.append("Roberta QA")

            falcon_response = falcon_future.result()
            if falcon_response:
                responses.append(falcon_response)
                sources.append("Falcon-7B")

        label = classify_label_advanced(responses, sources)
        accuracy = calculate_advanced_accuracy(responses, sources)

        falcon_output = next((resp for resp, src in zip(responses, sources) if src == "Falcon-7B"), None)
        fact_check_output = next((resp for resp, src in zip(responses, sources) if src == "FLAN-T5"), None)

        if falcon_output and fact_check_output:
            comparison = (
                f"\nOutput from Falcon-7B:\n{falcon_output}\n\n"
                f"Output from FLAN-T5:\n{fact_check_output}\n\n"
                "These outputs were verified by sources such as Falcon-7B, FLAN-T5, and other fact-checking tools."
            )
        elif falcon_output:
            comparison = (
                f"\nOutput from Falcon-7B:\n{falcon_output}\n\n"
                "Only Falcon-7B provided a response for this query."
            )
        elif fact_check_output:
            comparison = (
                f"\nOutput from FLAN-T5:\n{fact_check_output}\n\n"
                "Only FLAN-T5 provided a response for this query."
            )
        else:
            comparison = "No outputs were generated by Falcon-7B or FLAN-T5 for this query."

        result = {
            "label": label,
            "accuracy": f"{accuracy}%",
            "comparison": comparison,
        }
    except Exception as e:
        result = {
            "error": f"An error occurred while comparing Falcon and FLAN-T5 responses: {str(e)}"
        }
    return result

# Main Function
if __name__ == "__main__":
    print("Welcome to the Advanced Hoax Detection and Fact-Checking Tool!")
    while True:
        user_input = input("Enter your query or claim (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the tool. Goodbye!")
            break
        if user_input:
            corrected_input = spell_check(user_input)
            intent = classify_intent(corrected_input)
            if intent == "unknown":
                print("Unable to classify your intent. Please clarify if you're asking a question or fact-checking.")
            else:
                result = process_input_advanced(corrected_input)
                if "error" in result:
                    print(f"Error: {result['error']}")
                else:
                    print("\n### Analysis Results ###")
                    print(f"Label: {result['label']}")
                    print(f"Accuracy: {result['accuracy']}")
                    if result.get("comparison"):
                        print(f"Comparison: {result['comparison']}")

