"""Word embeddings from the same English model used in Module 2 Practical 3."""

import spacy

MODEL_NAME = "en_core_web_lg"


class EmbeddingModel:
    def __init__(self):
        # Load once at startup. Static word vectors need the tokenizer and
        # vocabulary, but not the parser, tagger, or named-entity recognizer.
        self.nlp = spacy.load(
            MODEL_NAME,
            exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"],
        )

    def calculate_embedding(self, word: str) -> dict:
        doc = self.nlp(word)
        if len(doc) != 1 or not doc[0].is_alpha:
            raise ValueError("Provide one alphabetic word, for example apple.")
        if not doc.has_vector:
            raise LookupError("This word has no vector in en_core_web_lg. Try another English word.")
        vector = doc.vector
        return {
            "word": word,
            "model": MODEL_NAME,
            "dimensions": len(vector),
            "embedding": vector.tolist(),
        }
