import numpy as np
import random, json, pickle, nltk, spacy, re, requests, pyowm
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from ava_config import ava_OWM_api_key

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




def get_weather(place):
    api_key = ava_OWM_api_key
    #base_url = "http://api.openweathermap.org/data/2.5/weather?"
    OpenWMap = pyowm.OWM(api_key)
    mgr = OpenWMap.weather_manager()
    weather = mgr.weather_at_place(place).weather

    dump_dict = weather.to_dict()
    '''
    {'reference_time': 1622875874, 
    'sunset_time': 1622862867, 
    'sunrise_time': 1622810480, 
    'clouds': 1, 
    'rain': {}, 
    'snow': {}, 
    'wind': {'speed': 4.99, 'deg': 321, 'gust': 10.11}, 'humidity': 39, 
    'pressure': {'press': 1011, 'sea_level': None}, 
    'temperature': {'temp': 298.22, 'temp_kf': None, 'temp_max': 301.28, 'temp_min': 294.82, 'feels_like': 297.8}, 
    'status': 'Clear', 
    'detailed_status': 'clear sky', 
    'weather_code': 800, 
    'weather_icon_name': '01n', 
    'visibility_distance': 10000, 
    'dewpoint': None, 
    'humidex': None, 
    'heat_index': None, 
    'utc_offset': -25200, 
    'uvi': None, 
    'precipitation_probability': None}
    '''
    curr_temp = str(round((int(dump_dict['temperature']['temp']) - 273.15) * 9 / 5 + 32)) + " °F"
    high_temp = str(round((int(dump_dict['temperature']['temp_max']) - 273.15) * 9 / 5 + 32)) + " °F"
    low_temp = str(round((int(dump_dict['temperature']['temp_min']) - 273.15) * 9 / 5 + 32)) + " °F"
    feels_like_temp = str(round((int(dump_dict['temperature']['feels_like']) - 273.15) * 9 / 5 + 32)) + " °F"
    
    print("The current temperature is " + curr_temp + " but it feels like " + feels_like_temp)
    print("Today's high is " + high_temp)
    print("Today's low is " + low_temp)

def get_weather_forecast_at_time(place, time):
    '''
    #time = '2020-08-03 16:30:00+00'
    api_key = ava_OWM_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    OpenWMap = pyowm.OWM(api_key)

    rain=forecast.will_be_rainy_at(time) # forecast rain
    sun=forecast.will_be_sunny_at(time) # forecast sun
    cloud=forecast.will_be_cloudy_at(time) # forecast clouds

    print("There will be rain :",rain) # print details
    print("There will be sun :",sun) #print details
    print("There will be clouds :",cloud) # print details
    '''

def return_named_entities(sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    ents = {}
    for ent in doc.ents:
        #print(ent.text, ent.start_char, ent.end_char, ent.label_)
        ents[ent.text] = ent.label_
    return ents

def get_city(dict):
    for key, value in dict.items():
        if value == 'GPE':
            print("key: ", key)
            return key
    # If no city is found, use user's city
    res = requests.get("https://ipinfo.io")
    # https://ipinfo.io/developers
    '''
        --example response--
    {   
        'ip': 'XX.XXX.XXX.XX',
        'hostname': 'hostname.host.name',
        'city': '[city name]',
        'region': '[state]',
        'country': '[2 char country code',
        'loc': '[lat,long]',
        'org': '[ISP]',
        'postal': '[zip code]',
        'timezone': 'America/Los_Angeles',
        'readme': 'https://ipinfo.io/missingauth'
    }
    
    '''
    print("res: ", res.json()['city'])
    return res.json()['city']


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
        city = get_city(return_named_entities(message))
        get_weather(city)




