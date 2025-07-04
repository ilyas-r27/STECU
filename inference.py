from sentence_transformers import SentenceTransformer, util
import torch
import json

# Load model SBERT Multilingual
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Load daftar minat dan karier dari JSON
with open('careers.json', 'r') as f:
    interest_to_career = json.load(f)

list_interest = list(interest_to_career.keys())

def analyze_user_text(text, top_k=2):
    emb_user = model.encode(text, convert_to_tensor=True)
    emb_interests = model.encode(list_interest, convert_to_tensor=True)

    cosine_scores = util.cos_sim(emb_user, emb_interests)[0]
    top_results = torch.topk(cosine_scores, k=top_k)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        interest = list_interest[idx]
        match_score = round(float(score), 2)
        careers = interest_to_career.get(interest, [])
        results.append({
            "interest": interest,
            "match_score": match_score,
            "career_recommendations": careers
        })

    return results

if __name__ == "__main__":
    with open("sample_input.txt", "r") as f:
        user_text = f.read()
    
    output = analyze_user_text(user_text)
    print(json.dumps(output, indent=2, ensure_ascii=False))

