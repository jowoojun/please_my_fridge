from pymongo import MongoClient
import argparse

from google.cloud import translate
import six

print("connecting DB")
client = MongoClient('ds119049.mlab.com', 19049)

print("connecting collections")
db = client['please_my_fridge']
db.authenticate(os.environ["mongoID"], os.environ["mongoPW"])
recipes = db.full_recipes

print("Starting Data Process")
for i, recipe in enumerate(recipes.find()):
    ap = "python google_images_download.py -k " + '"' + recipe["title"].encode("utf-8") + '"' + " -l 1"
    print(ap)
