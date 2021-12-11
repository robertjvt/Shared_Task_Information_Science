import nltk

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

    # Only take the first hypernym of the input word.
    hypernym = hypernyms[0]

    # Get the hypernym word of the input word.
    hyper = [str(lemma.name()) for lemma in hypernym.lemmas()][0]
    return hyper


def main():
    print(get_hypernym("dog"))



if __name__ == "__main__":
    main()
