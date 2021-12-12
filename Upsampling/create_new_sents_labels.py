import nltk
import random
from nltk.corpus import wordnet as wn

nltk.download('wordnet')


def get_hypernym(word):
    '''Retrieve hypernym of an input word from WordNet.'''

    # Retrieve synsets of word.
    synsets = wn.synsets(word, pos=wn.NOUN)

    # Only take the first synset of the input word.
    synset = synsets[0]

    # Retrieve hypernyms of the input word.
    hypernyms = synset.hypernyms()

    # Only take the first hypernym generator of the input word.
    hypernym = hypernyms[0]

    # Get the hypernym word of the input word.
    hyper = [str(lemma.name()) for lemma in hypernym.lemmas()][0]
    return hyper


def get_templates(file):
    '''Get templates containing single nouns
       from other_templates.txt'''

    singular_templates = []
    plural_templates = []
    with open(file, 'r') as infile:
        for line in infile:
            if '[PL-NOUN-' not in line:
                singular_templates.append(line)
            else:
                plural_templates.append(line)

    return singular_templates, plural_templates


def get_plural_noun(noun):
    '''Convert noun to plural form.'''

    # Put words which have a foreign origin in exception.
    # This list can be expanded.
    exception = ["taco", "avocado", "maestro"]

    for char in noun:
        if noun in exception:
            plural_noun = noun + "s"
        elif noun[-1] in "szjxo" or noun[-2:] in "shchzz":
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
    '''Check whether a character is a vowel.'''
    return char in "aeiou"


def is_consonant(char):
    '''Check whether a character is a consonant.'''
    return char in "bcdfghjklmnpqrstvwxyz"


def get_wordnet_noun():
    '''Get nouns from wordnet'''

    nouns = list(wn.all_synsets('n'))
    noun = random.choice(nouns).lemma_names()[0]

    return noun


def good_sentence_generator():
    pass


def false_sentence_generator():
    pass

def main():
    # print(get_hypernym("dog"))
    # print(get_plural_noun_templates('other_templates.txt'))
    # print(get_single_noun_templates('other_templates.txt'))

    # print(get_plural_noun("taco"))

    noun = get_wordnet_noun()
    print(noun)
    print(get_hypernym(noun))


if __name__ == "__main__":
    main()
