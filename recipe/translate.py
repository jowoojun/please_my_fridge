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
recipes = db.full_format_recipes

count = 0
for i, recipe in enumerate(recipes.find()):
    #if "title" in recipe:
    #    print(j ,"st, title : " ,MyPrettyPrinter().pprint(recipe['title']))
    
    title = recipe["title"].replace('"', '').replace('\n','').replace(',','').replace("`","'").encode("utf-8")
    recipe.update({"title": title}, upsert=False)
    recipes.save(recipe)

    #title = recipe["title"]
    #if "_id" in recipe:
    #    title = '"' + title + '"'
    #    print(title)
    #    recipe.update({"title": title[1:]}, upsert=False)
    #    #recipes.save(recipe)
    #if " " in title[-1]:
    #    title = '"' + title + '"'
    #    print(title)
    #    recipe.update({"title": title}, upsert=False)
    #    recipes.save(recipe)
    
    #recipe.update({"url": ""})
    #recipes.save(recipe)





    #print "googleimagesdownload -k \"", recipe["title"].encode("utf-8"),"\" -l 1"
    #ap = "googleimagesdownload -k " + '"' + recipe["title"].encode("utf-8") + '"' + " -l 1"
    #print(ap)



#print(count)

