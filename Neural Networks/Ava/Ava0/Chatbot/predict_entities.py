import spacy


'''
    ADJ: adjective
    ADP: adposition
    ADV: adverb
    AUX: auxiliary verb
    CONJ: coordinating conjunction
    DET: determiner
    INTJ: interjection
    NOUN: noun
    NUM: numeral
    PART: particle
    PRON: pronoun
    PROPN: proper noun
    PUNCT: punctuation
    SCONJ: subordinating conjunction
    SYM: symbol
    VERB: verb
    X: other
'''

def return_all_nouns(sentence):
    nlp = spacy.load('en_core_web_sm')
    sent = sentence
    doc = nlp(sent)
    allNoun_toks = [tok for tok in doc if (tok.pos_ == 'PROPN' or tok.pos_ == 'NOUN')]
    return allNoun_toks

def return_subject(sentence):
    nlp = spacy.load('en_core_web_sm')
    sent = sentence
    doc = nlp(sent)
    sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj")]
    return sub_toks

