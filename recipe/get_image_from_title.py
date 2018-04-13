from pymongo import MongoClient
import argparse

from google.cloud import translate
import six

client = MongoClient('localhost', 27017)

db = client['recipe_default_info']
recipes = db.full_format_recipes

#for i, recipe in enumerate(recipes.find()):

