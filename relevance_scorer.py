from sentence_transformers import SentenceTransformer

def main():
    # Load model once
    model = SentenceTransformer('all-mpnet-base-v2')

    while True:
        print("\n" + "="*50)
        print("SEMANTIC RELEVANCE CHECKER (Ctrl+C to exit)")
        print("="*50)
        
        # 1. Input your Target Keyword
        target_keyword = input("\nEnter Target Keyword: ")
        if not target_keyword: break
        
        # 2. Input the Text you wrote (Intro, Paragraph, etc.)
        print("\nPaste your text below (Press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "": break
            lines.append(line)
        content_text = "\n".join(lines)
        
        if not content_text: continue

        # 3. Calculate Score
        # We encode individually to compare 1-to-1
        keyword_emb = model.encode(target_keyword)
        content_emb = model.encode(content_text)
        
        score = model.similarity(keyword_emb, content_emb).item()
        
        # 4. Give Feedback
        print(f"\n>>> Semantic Score: {score:.4f}")
        
        if score > 0.6:
            print("✅ EXCELLENT: Highly relevant context.")
        elif score > 0.4:
            print("⚠️  OKAY: Somewhat relevant, but could be tighter.")
        else:
            print("❌ POOR: The text drifts away from the keyword topic.")

if __name__ == "__main__":
    main()