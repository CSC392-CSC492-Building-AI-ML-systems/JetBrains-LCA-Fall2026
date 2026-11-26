"""TF-IDF ranking adapted from the archived JetBrains baseline at 977d200d."""

import re
import string

import numpy as np
from nltk import PorterStemmer, WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CAMEL_COMPONENT = re.compile(r".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")


class TfIdfBaseline:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, text):
        tokens = []
        for token in word_tokenize(text):
            if token in string.punctuation:
                continue
            for component in CAMEL_COMPONENT.findall(token):
                term = self.lemmatizer.lemmatize(self.stemmer.stem(component.lower()))
                if term not in self.stop_words:
                    tokens.append(term)
        return tokens

    def localize_bugs(self, issue_description, repo_content):
        # The archive fits one vocabulary over the issue and the path-prefixed files.
        paths = np.asarray(list(repo_content), dtype=str)
        documents = [issue_description]
        documents.extend(f"{path}\n{content}" for path, content in repo_content.items())
        vectors = TfidfVectorizer(tokenizer=self.tokenize, token_pattern=None).fit_transform(documents)
        scores = cosine_similarity(vectors[0], vectors[1:])[0]
        # Preserve the archived NumPy tie ordering.
        order = np.argsort(scores)[::-1]
        return {"final_files": list(paths[order]), "rank_scores": list(scores[order])}
