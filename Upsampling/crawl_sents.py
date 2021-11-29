#!/usr/bin/env python

import enum
from pprint import pprint
import json

import spacy
from tqdm import tqdm


nlp = spacy.load(
    "en_core_web_md",
    disable=["textcat", "lemmatizer", "ner"],
)


def main():
    criteria = 1.0
    articles = load_articles()
    templates = load_templates()
    candidates = set()

    with open("candidate_sentences.txt", "w") as fd:
        with tqdm(total=len(articles)) as pbar:
            for doc in nlp.pipe(articles):
                sents_text = []
                sents_pos = []

                for sent in doc.sents:
                    sents_text.append(sent.text)
                    sents_pos.append(" ".join(token.pos_ for token in sent))

                sents_pos = [pos_doc for pos_doc in nlp.pipe(sents_pos)]

                for template in templates:
                    for st, sp in zip(sents_text, sents_pos):
                        if st in candidates:
                            continue
                        if not has_vector(sp):
                            continue
                        score_1 = 0.0
                        score_2 = 0.0
                        if has_vector(template["pos_1"]):
                            score_1 = template["pos_1"].similarity(sp)
                        if has_vector(template["pos_2"]):
                            score_2 = template["pos_2"].similarity(sp)
                        if score_1 >= criteria or score_2 >= criteria:
                            candidates.add(st)
                            fd.write(f"{st}\n")

                pbar.update(1)

    print(len(candidates))


def load_articles():
    class State(enum.Enum):
        read_title = 1
        read_article = 2

    state = State.read_title
    articles = []

    with open("corpus.txt", "r") as fd:
        for line in fd:
            if state == State.read_title:
                state = State.read_article
                articles.append("")

            elif state == State.read_article:
                if line.strip() == "":
                    # Remove space from the beginning of the article.
                    articles[-1] = articles[-1].lstrip()
                    state = State.read_title
                else:
                    # Put space before text to keep sentences separated.
                    # And strip whitespace (e.g. linebreaks) from text
                    # chunks.
                    articles[-1] += f" {line.strip()}"

    return articles


def load_templates():
    templates = []

    for template in tqdm(TEMPLATES):
        doc = nlp(template)
        pos = [token.pos_ for token in doc]
        pos_1 = nlp(" ".join(pos))
        pos_2 = nlp(" ".join(["NOUN", *pos[1:]]))
        templates.append({
            "text": doc.text,
            "pos_1": pos_1,
            "pos_2": pos_2,
        })

    return templates


def process_article(article):
    sentences = []

    doc = nlp(article)
    for sent in doc.sents:
        # print(f"{sent.text!r}")
        sentences.append({
            "obj": sent,
            "text": sent.text,
            "pos": nlp(" ".join(token.pos_ for token in sent)),
        })

    return sentences


def has_vector(doc):
    """Check if a sentence can be compared with Spacy.

    A sentence like "kasimov" can't be compared.
    """
    return all(sent.has_vector for sent in doc.sents)



TEMPLATES = [
    "I like word , but not word .",
    "I like word , and word too .",
    "I like word more than word .",
    "I like word , an interesting type of word .",
    "I do not like word , I prefer word .",
    "I like word , and more specifically word .",
    "I like word , except word .",
    "I use word , except word .",
    "He likes word , and word too .",
    "I use word , and more specifically word .",
    "I met word , and more specifically word .",
    "He trusts his word , an interesting type of word .",
    "He trusts his word , except word .",
    "I use word , and word too .",
    "He trusts word , and more specifically his word .",
    "He likes word , an interesting type of word .",
    "He does not trust word , he prefers his word .",
    "I met word , an interesting type of word .",
    "He does not like word , he prefers word .",
    "He trusts his word more than word .",
    "He trusts word , an interesting type of word .",
    "He trusts his word , but not word .",
    "I use word more than word .",
    "I use word , but not word .",
    "I met word , and word too .",
    "I use word , an interesting type of word .",
    "He likes word more than word .",
    "He does not trust his word , he prefers word .",
    "I met word , except word .",
    "I met word , but not word .",
    "He trusts his word , and word too .",
    "He likes word , but not word .",
    "He trusts his word , and more specifically word .",
    "He trusts word , except his word .",
    "He likes word , and more specifically word .",
    "He likes word , except word .",
    "He trusts word more than his word .",
    "He trusts word , but not his word .",
    "He trusts his word , and his word too .",
    "He trusts word , and his word too .",
]


if __name__ == "__main__":
    main()
