# -*- coding: utf-8 -*-

from pymongo import MongoClient
import os
from sklearn.metrics import mean_squared_error

client = MongoClient('ds119049.mlab.com', 19049)

print("connecting collections")
db = client['please_my_fridge']
db.authenticate(os.environ["mongoID"], os.environ["mongoPW"])
recipes = db.full_recipes

print("Starting Data Process")
recipe_category_id = []

recipe_info_li = []

for i, recipe in enumerate(recipes.find()):
    recipe_info = [recipe["title"], recipe["categories"], recipe["ingredients"], recipe["directions"]]
    recipe_info_li.append(recipe_info)

for i, recipe in enumerate(recipes.find()):
    try:
        recipe_id = recipe["_id"]
        recipe_categories = recipe["categories"]
        lowered_category = ""
        for category in recipe_categories:
            category = category.lower().replace(" ","_")
            if lowered_category == "":
                lowered_category = category
            else:
                lowered_category = lowered_category + ", " + category
        recipe_category_id.append(lowered_category)

    except ValueError:
        recipe_category_id.append('')
print(len(recipe_category_id))

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=3, stop_words='english')
X = vectorizer.fit_transform(recipe_category_id[:100])
# TF-IDF로 변환한 키워드의 리스트
# X의 0번 열에 해당하는 키워드가 feature_names[0]의 키워드입니다.
feature_names = vectorizer.get_feature_names()
print(feature_names)

from sklearn.metrics.pairwise import cosine_similarity
recipe_sim = cosine_similarity(X)
def similar_recommend_by_movie_id(recipe_category_id, recipes_id):
    recipe_index = recipes_id - 1
    # enumerate 함수로 [(리스트 인덱스 0, 유사도 0), (리스트 인덱스 1, 유사도 1)...]의
    # 리스트를 만듭니다. 그 후 각 튜플의 두 번째 항목, 즉 유사도를 이용하여 내림차순 정렬합니다.
    # 이렇게 만든 리스트의 가장 앞 튜플의 첫 번째 항목이 영화 ID가 됩니다.
    similar_recipes = sorted(list(enumerate(recipe_sim[recipe_index])),key=lambda x:x[1], reverse=True)
    recommended = 1
    index = recipe_info_li[recipes_id]
    print("-----recommendation for recipe ------",index[0])
    for recipe_info in similar_recipes[1:6]:
        # 주어진 영화와 가장 비슷한 영화는 그 영화 자신이므로 출력 시 제외합니다.
        recipe_title = recipe_info_li[recipe_info[0]]
        print('rank %d recommendation:%s'%(recommended, recipe_title[0]))
        #print(recipe_title[1])
        recommended+=1

similar_recommend_by_movie_id(recipe_category_id, 1)
