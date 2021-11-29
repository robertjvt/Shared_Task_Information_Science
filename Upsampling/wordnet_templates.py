from nltk.corpus import wordnet as wn


def read_templates():
    # Returns only regular nouns
    templates = []
    f = open('other_templates.txt', 'r')
    data = f.readlines()
    for line in data:
        if '[PL-NOUN]' not in line:
            templates.append(line[:-3])
    f.close()
    return templates


def wordnet_word(word):
    synsets = wn.synsets(word, pos=wn.NOUN)
    synset = synsets[0]
    hypernyms = synset.hypernyms()
    hypernym = hypernyms[0]
    words = [str(lemma.name()) for lemma in hypernym.lemmas()]
    return words[0]


def inject_words(word, new_word, templates):
    for template in templates:
        template = template.split()
        template[0] = word
        template[-1] = new_word
        print(template)


def main():
    templates = read_templates()

    # Take a word from provided templates (currently fixed)
    word = 'pig'

    # Take the WordNet hypernym word
    new_word = wordnet_word(word)
    print(new_word)

    # Inject the new words into the templates
    inject_words(word, new_word, templates)


if __name__ == '__main__':
    main()