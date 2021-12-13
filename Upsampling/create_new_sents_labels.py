import nltk
import random
import os
from nltk.corpus import wordnet as wn


# nltk.download("wordnet")

DATA_DIR = '../Data/split_dataset/'
NOUNS = list(wn.all_synsets("n"))
RANGE = 50

def get_hypernym(word):
    """Retrieve hypernym of an input word from WordNet."""

    # Retrieve synsets of word.
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return None

    # Only take the first synset of the input word.
    synset = synsets[0]

    # Retrieve hypernyms of the input word.
    hypernyms = synset.hypernyms()
    if not hypernyms:
        return None

    # Only take the first hypernym generator of the input word.
    hypernym = hypernyms[0]

    # Get the hypernym word of the input word.
    hyper = [str(lemma.name()) for lemma in hypernym.lemmas()][0]

    return hyper


def get_templates(file):
    """Get templates containing single nouns
       from other_templates.txt"""

    singular_templates = []
    plural_templates = []
    with open(file, "r") as infile:
        for line in infile:
            if "[PL-NOUN-" not in line:
                singular_templates.append(line)
            else:
                plural_templates.append(line)

    return singular_templates, plural_templates


def get_plural_noun(noun):
    """Convert noun to plural form."""

    # Put words which have a foreign origin in exception.
    # This list can be expanded.
    exception = ["taco", "avocado", "maestro"]

    noun = str(noun)
    for char in noun:
        if noun in exception:
            plural_noun = noun + "s"
        elif noun[-1] in "szjxo" or noun[-2:] in "shchzz":
            plural_noun = noun + "es"
        elif noun[-2:] == "cy" or noun[-2:] == "ty":
            noun = noun.replace(noun[-1], "i")
            plural_noun = noun + "es"
        elif noun[-1] == "y" and is_vowel(noun[-2]):
            plural_noun = noun + "s"
        elif noun[-1] == "y" and is_consonant(noun[-2]):
            plural_noun = noun + "es"
        elif noun[-1] == "f":
            noun = noun.replace(noun[-1], "v")
            plural_noun = noun + "es"
        elif noun[-2:] == "fe":
            noun = noun.replace(noun[-2], "v")
            plural_noun = noun + "s"
        else:
            plural_noun = noun + "s"

        return plural_noun


def is_vowel(char):
    """Check whether a character is a vowel."""
    return char in "aeiou"


def is_consonant(char):
    """Check whether a character is a consonant."""
    return char in "bcdfghjklmnpqrstvwxyz"


def get_wordnet_noun():
    """Get nouns from wordnet"""

    noun = random.choice(NOUNS).lemma_names()[0]

    return noun


def get_negative_template(sentence):
    """Get negative form of verb."""
    sentence = sentence.split()
    if not sentence:
        return None

    be_verbs = ["is", "are", "was", "were"]

    if sentence[1] not in be_verbs:
        if sentence[1][-1] == "s":
            sentence[1] = sentence[1][:-1]
            sentence.insert(1, "does")
            sentence.insert(2, "not")

        elif sentence[1][-2:] == "ed":
            sentence[1] = sentence[1][:-1]
            sentence.insert(1, "did")
            sentence.insert(2, "not")
        else:
            sentence.insert(1, "do")
            sentence.insert(2, "not")

    else:
        sentence.insert(2, "not")

    return sentence


def good_sentence_generator(file):
    """Create good sentences by:
    - injecting noun and its hypernyms in the good place,
    - or making templates negative and injecting a pair of
    words having no taxonomy relation to the sentences.
    Return a list of good sentences with label 1."""

    # Get templates from file.
    singular_templates, plural_templates = get_templates(file)

    sentences = []

    # Create new singular sentences by injecting noun and
    # its hypernyms in the good place.
    for st in singular_templates:
        st = st.split()
        if not st:
            continue

        for _ in range(RANGE):
            while True:
                noun = get_wordnet_noun()
                hypernym = get_hypernym(noun)
                if hypernym:
                    break

            st2 = st[:]
            if st[0] == "[NOUN-HYPO]":
                st2[0] = noun
                st2[-2] = hypernym
            elif st[0] == "[NOUN-HYPER]":
                st2[0] = hypernym
                st2[-2] = noun

            # Add label 1.
            st2 += ["1"]
            sentences.append(" ".join(st2))


    # Create new plural sentences by injecting noun and
    # its hypernyms in the good place.
    for pt in plural_templates:
        pt = pt.split()
        if not pt:
            continue

        for _ in range(RANGE):
            while True:
                noun = get_wordnet_noun()
                hypernym = get_hypernym(noun)
                if hypernym:
                    break
            # Get plural form of noun
            noun = get_plural_noun(noun)
            hypernym = get_plural_noun(hypernym)
            pt2 = pt[:]
            if pt[0] == "[PL-NOUN-HYPO]":
                pt2[0] = noun
                pt2[-2] = hypernym
            elif pt[0] == "[PL-NOUN-HYPER]":
                pt2[0] = hypernym
                pt2[-2] = noun

            # Add label 1.
            pt2 += ["1"]
            sentences.append(" ".join(pt2))


    # Create new singular sentences by making template negative
    # and injecting a pair of words having not taxonomy relation.

    for st in singular_templates:
        # Get negative template
        st = get_negative_template(st)
        if not st:
            continue

        for _ in range(RANGE):
            noun_1 = get_wordnet_noun()
            noun_2 = get_wordnet_noun()

            while noun_1 == noun_2:
                noun_2 = get_wordnet_noun()

            st2 = st[:]
            st2[0] = noun_1
            st2[-2] = noun_2
            st2 += ["1"]
            sentences.append(" ".join(st2))


    # Create new plural sentences by by making template negative
    # and injecting a pair of words having not taxonomy relation.

    for pt in plural_templates:
        # Get negative template
        pt = get_negative_template(pt)
        if not pt:
            continue

        for _ in range(RANGE):
            noun_1 = get_plural_noun(get_wordnet_noun())
            noun_2 = get_plural_noun(get_wordnet_noun())

            while noun_1 == noun_2:
                noun_2 = get_wordnet_noun()

            pt2 = pt[:]
            pt2[0] = noun_1
            pt2[-2] = noun_2
            pt2 += ["1"]
            sentences.append(" ".join(pt2))

    return sentences



def false_sentence_generator(file):
    """Create false sentences by:
    - inverting noun and its hypernyms in the sentences,
    - or making templates negative then injecting a pair
    hyper-hypo in the good place,
    - or injecting a pair of words having no taxonomy relation
    to the sentences.
    Return a list of false sentences with label 0."""

    # Get templates from file.
    singular_templates, plural_templates = get_templates(file)

    sentences = []

    # Create new singular sentences by inverting noun and its
    # hypernyms in the sentences.
    for st in singular_templates:
        st = st.split()
        if not st:
            continue

        for _ in range(RANGE):
            while True:
                noun = get_wordnet_noun()
                hypernym = get_hypernym(noun)
                if hypernym:
                    break

            st2 = st[:]
            if st[0] == "[NOUN-HYPO]":
                st2[0] = hypernym
                st2[-2] = noun
            elif st[0] == "[NOUN-HYPER]":
                st2[0] = noun
                st2[-2] = hypernym

            # Add label 0.
            st2 += ["0"]
            sentences.append(" ".join(st2))


    # Create new plural sentences by inverting noun and its
    # hypernyms in the sentences
    for pt in plural_templates:
        pt = pt.split()
        if not pt:
            continue

        for _ in range(RANGE):
            while True:
                noun = get_wordnet_noun()
                hypernym = get_hypernym(noun)
                if hypernym:
                    break

            # Get plural form of nouns.
            noun = get_plural_noun(noun)
            hypernym = get_plural_noun(hypernym)
            pt2 = pt[:]
            if pt[0] == "[PL-NOUN-HYPO]":
                pt2[0] = hypernym
                pt2[-2] = noun
            elif pt[0] == "[PL-NOUN-HYPER]":
                pt2[0] = noun
                pt2[-2] = hypernym

            # Add label 0.
            pt2 += ["0"]
            sentences.append(" ".join(pt2))


    # Create new singular sentences by making template negative
    # then injecting a pair of hyper-hypo in the good position.

    for st in singular_templates:
        # Get negative template
        st = get_negative_template(st)
        if not st:
            continue

        for _ in range(RANGE):
            while True:
                noun = get_wordnet_noun()
                hypernym = get_hypernym(noun)
                if hypernym:
                    break

            st2 = st[:]
            if st[0] == "[NOUN-HYPO]":
                st2[0] = noun
                st2[-2] = hypernym
            elif st[0] == "[NOUN-HYPER]":
                st2[0] = hypernym
                st2[-2] = noun

            # Add label 0.
            st2 += ["0"]
            sentences.append(" ".join(st2))


    # Create new plural sentences by making template negative
    # then injecting a pair of hyper-hypo in the good position.

    for pt in plural_templates:
        # Get negative template
        pt = get_negative_template(pt)
        if not pt:
            continue

        for _ in range(RANGE):
            while True:
                noun = get_wordnet_noun()
                hypernym = get_hypernym(noun)
                if hypernym:
                    break

            noun = get_plural_noun(noun)
            hypernym = get_plural_noun(hypernym)
            pt2 = pt[:]
            if pt[0] == "[PL-NOUN-HYPO]":
                pt2[0] = noun
                pt2[-2] = hypernym
            elif pt[0] == "[PL-NOUN-HYPER]":
                pt2[0] = hypernym
                pt2[-2] = noun

            # Add label 0.
            pt2 += ["0"]
            sentences.append(" ".join(pt2))

    # Create new singular sentences by injecting a pair
    #of words having no taxonomy relation to the sentences.

    for st in singular_templates:
        st = st.split()
        if not st:
            continue

        for _ in range(RANGE):
            noun_1 = get_wordnet_noun()
            noun_2 = get_wordnet_noun()

            while noun_1 == noun_2:
                noun_2 = get_wordnet_noun()

            st2 = st[:]
            st2[0] = noun_1
            st2[-2] = noun_2
            st2 += ["0"]
            sentences.append(" ".join(st2))


    # Create new plural sentences by injecting a pair
    #of words having no taxonomy relation to the sentences.

    for pt in plural_templates:
        pt = pt.split()
        if not pt:
            continue

        for _ in range(RANGE):
            noun_1 = get_plural_noun(get_wordnet_noun())
            noun_2 = get_plural_noun(get_wordnet_noun())

            while noun_1 == noun_2:
                noun_2 = get_wordnet_noun()

            pt2 = pt[:]
            pt2[0] = noun_1
            pt2[-2] = noun_2
            pt2 += ["0"]
            sentences.append(" ".join(pt2))

    return sentences


def main():
    infile = "other_templates.txt"
    false_sentences = false_sentence_generator(infile)
    good_sentences = good_sentence_generator(infile)
    sentences = false_sentences + good_sentences

    # Eliminate the under steep between wordnet nouns.
    cleaned_sentences = [sent.replace("_", " ") for sent in sentences]

    random.shuffle(cleaned_sentences)

    outfile_path = DATA_DIR + "data_generated_from_other_templates.txt"
    with open(outfile_path,"w") as outfile:
        for sent in cleaned_sentences:
            outfile.write(sent + "\n")


if __name__ == "__main__":
    main()
