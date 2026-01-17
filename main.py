from src.text_loader import load_text
from src.similarity import calculate_similarity
from src.concept_mapper import map_concepts

def main():
    eu_text = load_text("data/eu_ai_act_excerpt.txt")
    tr_text = load_text("data/kvkk_excerpt.txt")

    score = calculate_similarity(eu_text, tr_text)
    concepts = map_concepts()

    print(f"Similarity score: {score:.2f}")
    for key, values in concepts.items():
        print(f"{key}: {values}")

if __name__ == "__main__":
    main()