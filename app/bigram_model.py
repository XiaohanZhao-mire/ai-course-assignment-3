"""Adapted from Module 2 Practical 2: Word Sampling."""

import random
import re
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus: list[str]):
        counts: dict[str, Counter] = defaultdict(Counter)
        # Each sample is a separate sequence: do not join unrelated sentences.
        for text in corpus:
            words = re.findall(r"\b\w+\b", text.lower())
            for current, following in zip(words, words[1:]):
                counts[current][following] += 1
        self.probabilities = {
            word: {following: count / sum(next_words.values())
                   for following, count in next_words.items()}
            for word, next_words in counts.items()
        }

    def generate_text(self, start_word: str, length: int) -> str:
        """Return at most length words, including the starting word."""
        words = [start_word.lower()]
        for _ in range(length - 1):
            choices = self.probabilities.get(words[-1])
            if not choices:
                break
            words.append(random.choices(list(choices), weights=list(choices.values()))[0])
        return " ".join(words)
