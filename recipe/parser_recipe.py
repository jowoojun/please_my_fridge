import json
from pprint import pprint

with open('../document_recipe/recipe_process_info.json') as data_file:
    data = json.load(data_file)

pprint(data.encode("utf-8"))
