import pprint
from pymongo import MongoClient
import argparse

from google.cloud import translate
import six
import os

import pprint
class MyPrettyPrinter(pprint.PrettyPrinter):
    def format(self, _object, context, maxlevels, level):
        if isinstance(_object, unicode):
            return "'%s'" % _object.encode('utf8'), True, False
        elif isinstance(_object, str):
            _object = unicode(_object,'utf8')
            return "'%s'" % _object.encode('utf8'), True, False
        return pprint.PrettyPrinter.format(self, _object, context, maxlevels, level)
print("connecting DB")
client = MongoClient('ds119049.mlab.com', 19049)

print("connecting collections")
db = client['please_my_fridge']
db.authenticate(os.environ["mongoID"], os.environ["mongoPW"])
recipes = db.full_recipes

print("Starting Data Process")
for i, recipe in enumerate(recipes.find()):
    """
    title = recipe["title"].replace('"', '').replace('\n','').replace(',','').encode("utf-8")
    recipe.update({"title": title}, upsert=False)
    recipes.save(recipe)
    """
    """
    title = recipe["title"]
    if "`" in title:
        title = title.replace("`", "'")
        recipe.update({"title": title}, upsert=False)
        recipes.save(recipe)
    """
    """
    recipe.update({"url": ""})
    recipes.save(recipe)
    """




