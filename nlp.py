import json
from fuzzywuzzy import process

def load_qa(path="qa_data.json"):
    """Load the question→answer dictionary."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def match_question(question, qa_pairs, threshold=60):
    """
    Find the best matching question in qa_pairs for the given text.
    Returns the corresponding answer, or the 'default' answer.
    """
    # list of keys (questions)
    questions = list(qa_pairs.keys())
    best_match, score = process.extractOne(question, questions)
    if score >= threshold and best_match in qa_pairs:
        return qa_pairs[best_match]
    return qa_pairs.get("default", "Sinabyumva neza.")

if __name__ == "__main__":
    # quick test
    qa = load_qa()
    for q in ["amakuru yawe?", "uraho neza!", "ukunda iki?"]:
        print(q, "→", match_question(q, qa))
