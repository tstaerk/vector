from gensim.models import KeyedVectors
import numpy as np

# Load pre-trained word embeddings (Google's Word2Vec embeddings in this example)
def load_embeddings():
    print("Loading word embeddings...")


    import gensim.downloader as api

    word_vectors = api.load('word2vec-google-news-300')
    #model_path = "path/to/GoogleNews-vectors-negative300.bin.gz"  # Update the path
    #word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Embeddings loaded successfully!")
    return word_vectors

# Calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Main function
def main():
    word_vectors = load_embeddings()

    while True:
        print("\nEnter two words to compute their similarity (or type 'exit' to quit):")
        word1 = input("Word 1: ").strip()
        word2 = input("Word 2: ").strip()

        if word1.lower() == "exit" or word2.lower() == "exit":
            print("Exiting program.")
            break

        if word1 in word_vectors and word2 in word_vectors:
            vec1 = word_vectors[word1]
            vec2 = word_vectors[word2]
            similarity = cosine_similarity(vec1, vec2)
            print(f"\nSimilarity between '{word1}' and '{word2}': {similarity:.4f}")
        else:
            print("\nOne or both words are not in the vocabulary. Please try different words.")

if __name__ == "__main__":
    main()

