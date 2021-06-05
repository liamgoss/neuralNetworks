import spacy, re
'''
    --Parts of Speech--
    ADJ:        adjective
    ADP:        adposition
    ADV:        adverb
    AUX:        auxiliary verb
    CONJ:       coordinating conjunction
    DET:        determiner
    INTJ:       interjection
    NOUN:       noun
    NUM:        numeral
    PART:       particle
    PRON:       pronoun
    PROPN:      proper noun
    PUNCT:      punctuation
    SCONJ:      subordinating conjunction
    SYM:        symbol
    VERB:       verb
    X:          other
    SPACE:      space

    --Syntactic Dependencies--
    nsubj:      nominal subject
    aux:        auxiliary
    ROOT:       {no explanation available}
    prep:       prepositional modifier
    pcomp:      complement of preposition
    compound:   compound
    dobj:       direct object
    quantmod:   modifier of quantifier
    pobj:       object of preposition


    --Detailed Part of Speech Tag--
    NNP:        noun, proper singular
    VBZ:        verb, 3rd person singular present
    VBG:        verb, gerund or present participle
    IN:         conjunction, subordinating or preposition
    NN:         noun, singular or mass
    CD:         cardinal number
'''



#                                       ---------COMPLETED---------
# Okay so after getting the pobj, see if there are any ADV after it (and concatenate them?) to provide more context
# Such as "...time in Fresno right now" - right now being two ADV's according to spacy (use indices to determine position)
# If word immediately following "in" (prep) is NOT pobj, but there is a pobj after it, try the substring in title case??
# Example, New York City shows up as three pronouns, but new york city does not and causes the pobj to be "city"
    # still happens without .lower(), so maybe if pobj in concatenated PROPN following prep?
#                                       ---------COMPLETED---------

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


#sentence = "What is the weather in New York City?"
sentence = "What is the weather in Fresno right now?"
#sentence = "What is the weather in New York City right now?"



def return_context(sentence):
    sentence_pobj, sentence_adv = provide_context(sentence.lower())
    print("pobj: ", sentence_pobj)
    if sentence_adv == '':
        print("No adverbs")
    else:
        print("adv: ", sentence_adv)

return_context(sentence)

'''
for type in depList:
    print(type + ": ", return_synt_dep(sentence, type))

print('-------------')

for type in posList:
    print(type + ": ", return_pos_type(sentence, type))
'''

'''
tagList = ['NNP', 'VBZ', 'VBG', 'IN', 'NN', 'CD']
def return_pos_tag(sentence, tag):
    nlp = spacy.load('en_core_web_sm')
    sent = sentence
    doc = nlp(sent)
    toks = [tok for tok in doc if (tok.tag_ == tag)]
    return toks
print('-------------')
for type in tagList:
    print(type + ": ", return_pos_tag(sentence.lower(), type))
'''