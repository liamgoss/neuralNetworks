import numpy as np
import random, json, pickle, nltk, spacy, re
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
#intents_model = load_model('chatbot.model')
intents_model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = intents_model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


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

def return_proper_nouns(sentence):
    nlp = spacy.load('en_core_web_sm')
    sent = sentence
    doc = nlp(sent)
    propNoun_toks = [tok for tok in doc if (tok.pos_ == 'PROPN')]
    return propNoun_toks

def return_nouns(sentence):
    nlp = spacy.load('en_core_web_sm')
    sent = sentence
    doc = nlp(sent)
    noun_toks = [tok for tok in doc if (tok.pos_ == 'NOUN')]
    return noun_toks

def return_subject(sentence):
    nlp = spacy.load('en_core_web_sm')
    sent = sentence
    doc = nlp(sent)
    sub_toks = [tok for tok in doc if (tok.dep_ == "nsubj")]
    return sub_toks



def provide_context(sentence, *args):
    posList = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
               'SCONJ', 'SYM', 'X']
    depList = ['nsubj', 'aux', 'ROOT', 'prep', 'pcomp', 'compound', 'dobj', 'quantmod', 'pobj']

    def return_pobjs(sentence):
        nlp = spacy.load('en_core_web_sm')
        sent = sentence
        doc = nlp(sent)
        pobj_toks = [tok for tok in doc if (tok.dep_ == 'pobj')]
        return pobj_toks

    def return_advs(sentence):
        nlp = spacy.load('en_core_web_sm')
        sent = sentence
        doc = nlp(sent)
        adv_toks = [tok for tok in doc if (tok.pos_ == 'ADV')]
        return adv_toks

    def return_preps(sentence):
        nlp = spacy.load('en_core_web_sm')
        sent = sentence
        doc = nlp(sent)
        prep_toks = [tok for tok in doc if (tok.dep_ == 'prep')]
        return prep_toks

    def next_word(target, source):
        for i, w in enumerate(source):
            if w == target:
                return source[i + 1]

    #print("argument is: ", sentence)
    prep = return_preps(sentence.lower())
    pobj = return_pobjs(sentence.lower())
    adv = return_advs(sentence.lower())
    adverbs = ''
    noAdv = False
    if not prep:
        #print("No prepositions")
        pass
    else:
        if not pobj:
            #print("No object(s) of preposition")
            pass

        else:
            pobj_index = sentence.index(str(pobj[0]))
            if not adv:
                #print("No adverb(s)")
                noAdv = True
            else:
                advs_after = []
                for adverb in adv:
                    curr_index = sentence.index(str(adverb))
                    if curr_index > pobj_index:
                        #print("appending adverb...")
                        advs_after.append(str(adverb))

            if not noAdv:

                runs = 0
                for adverb in advs_after:
                    ind = advs_after.index(adverb)
                    adverbs += adverb
                    adverbs += " "

                    if runs != 0:
                        #print(str(advs_after[ind+1]).lower())
                        try:
                            if re.sub(r'[^\w\s]', '', str(next_word(adverb, sentence.lower().split()))) == str(advs_after[ind+1]).lower():
                                adverbs += adverb
                                adverbs += " "
                            else:
                                pass
                        except:
                            pass

            #print("adverbs: ", adverbs)
            target = str(prep[0])
            split_sentence = sentence.lower().split()

            if next_word(target, split_sentence) != str(pobj[0]):
                # try title case on substring
                after_prep_sentence = sentence.split(str(prep[0]))[1]
                return provide_context(after_prep_sentence.title(), adverbs)

    #print("strip: ", sentence.strip())
    if sentence.strip() == sentence.strip().title():
        # We are in the recursive run
        advCheck = return_advs(sentence.lower())
        final_pobj = re.sub(r'[^\w\s]', '', sentence.strip())  # Remove punctuation using regex


        if advCheck:
            advCheckFirst = str(advCheck[0])

            if advCheckFirst.title() in final_pobj:
                #print('match')
                final_pobj = str(final_pobj.split(advCheckFirst.title())[0])
                advCheckString = ' '.join(map(str, advCheck))
                #advCheckString = advCheckString.title().strip()

                #final_pobj.strip().replace(advCheckString, '')
                return final_pobj, advCheckString
        return final_pobj, adverbs  # sentence (without whitespace) is our pobj
    else:

        return str(pobj[0]), adverbs


def return_context(sentence):
    sentence_pobj, sentence_adv = provide_context(sentence.lower())
    print("pobj: ", sentence_pobj)
    if sentence_adv == '':
        print("No adverbs")
    else:
        print("adv: ", sentence_adv)







print("Ava is running!")
while True:
    message = input("")
    ints = predict_class(message.lower())
    print("ints: ", ints)

    res = get_response(ints, intents)
    print(res)
    if ints[0]['intent'] == 'goodbye':
        exit(0)

    elif ints[0]['intent'] == 'weather':
        print("Subject is: ", return_subject(message))
        #return_context(message)