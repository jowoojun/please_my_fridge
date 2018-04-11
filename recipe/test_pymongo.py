import pprint
from pymongo import MongoClient
import argparse

from google.cloud import translate
import six

import pprint
class MyPrettyPrinter(pprint.PrettyPrinter):
    def format(self, _object, context, maxlevels, level):
        if isinstance(_object, unicode):
            return "'%s'" % _object.encode('utf8'), True, False
        elif isinstance(_object, str):
            _object = unicode(_object,'utf8')
            return "'%s'" % _object.encode('utf8'), True, False
        return pprint.PrettyPrinter.format(self, _object, context, maxlevels, level)
client = MongoClient('localhost', 27017)

db = client['recipe_default_info']
recipe = db.full_format_recipes


def translate_text(target, text):
    """Translates text into the target language.
    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    translate_client = translate.Client()
    del text['_id']
    del text['calories']
    del text['date']
    del text['desc']
    del text['fat']
    del text['protein']
    del text['rating']
    del text['sodium']
    

    text = text
    #text = text.decode('utf-8')
        
    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    new_dict = {}
    for key, value in text.items():
        if type(value) == list:
            text = []
            for i in range(len(value)):
                result = translate_client.translate(value[i], target_language=target)
                #print(u'Text: {}'.format(result['input']))
                #print(u'Translation: {}'.format(result['translatedText']))
                #print(u'Detected source language: {}'.format(result['detectedSourceLanguage']))
                text.append(result['translatedText'])
            new_dict[key] = text
                
        else:
            result = translate_client.translate(value, target_language=target)
            #print(u'Text: {}'.format(result['input']))
            #print(u'Translation: {}'.format(result['translatedText']))
            #print(u'Detected source language: {}'.format(result['detectedSourceLanguage']))
            new_dict[key] = result['translatedText']

    return new_dict

new_dic = translate_text('ko', recipe.find_one())
MyPrettyPrinter().pprint(new_dic)
