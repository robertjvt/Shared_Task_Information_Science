#!/usr/bin/env python

import spacy


nlp = spacy.load("en_core_web_md")


def main():
    with open("candidate_sentences.txt", "r") as fd_in, \
            open("unfiltered_templates.txt", "w") as fd_out:
        templates = []
        for doc in nlp.pipe(fd_in):
            for sent in doc.sents:
                verbs = [t for t in sent if t.pos_ == "VERB"]
                for verb in verbs:
                    if verb.text == verb.lemma_:
                        template = f"[PL-NOUN] {verb} [PL-NOUN] . \n"
                    else:
                        template = f"[NOUN] {verb} [NOUN] . \n"
                    if template not in templates:
                        templates.append(template)
                        fd_out.write(template)


if __name__ == "__main__":
    main()
