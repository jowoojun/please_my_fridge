# -*- coding: utf-8 -*-

from pymongo import MongoClient
import os
import numpy as np
from sklearn.metrics import mean_squared_error

#client = MongoClient('ds119049.mlab.com', 19049)
client = MongoClient('localhost', 27017)

print("connecting collections")
db = client['please_my_fridge']
db.authenticate(os.environ["mongoID"], os.environ["mongoPW"])
recipes = db.full_recipes
#db = client['recipe_default_info']
recipes = db.full_format_recipes
#recipes = db.fake_recipe_data
#users = db.fake_user_data


print("Starting Data Process")
recipe_title = []
recipe_categories = []
recipe_ingredients = []
recipe_directions = []
for i, recipe in enumerate(recipes.find()):
    recipe_title.append(recipe["title"])
    recipe_categories.append(recipe["categories"])
    recipe_ingredients.append(recipe["ingredients"])
    recipe_directions.append(recipe["directions"])

user_nick_name = []
user_tried = []
for i, user in enumerate(users.find()):
    user_nick_name.append(user["nickname"])
    user_tried.append(user["scrap"])


from sklearn import preprocessing
recipe_label_encoder = preprocessing.LabelEncoder()
recipe_label_encoder.fit(recipe_title)
encoded_recipe_title = np.array(recipe_label_encoder.transform(recipe_title), dtype=np.int32)
total_recipe_depth = encoded_recipe_title.max() + 1

nickname_label_encoder = preprocessing.LabelEncoder()
nickname_label_encoder.fit(user_nick_name)
encoded_user_nick_name = np.array(nickname_label_encoder.transform(user_nick_name), dtype=np.int32)
total_user_depth = encoded_user_nick_name.max() + 1

def read_rating_data():
    Q = np.zeros((total_user_depth, total_recipe_depth), dtype=np.float64)
    
    for i, nick in enumerate(encoded_user_nick_name):
        row = nick
        for j, tr in enumerate(user_tried[nick]):
            col = recipe_label_encoder.transform([tr])
            Q[row, col[0]] = float(5.0)

    return Q

R = read_rating_data()

from scipy import stats
user_mean_li = []
for i in range(0, R.shape[0]):
    user_rating = [x for x in R[i] if x>0.0]
    user_mean_li.append(stats.describe(user_rating).mean)

print(stats.describe(user_mean_li))

recipe_mean_li = []
for i in range(0, R.shape[1]):
    R_T = R.T
    print(R_T)
    recipe_rating = [x for x in R_T[i] if x>0.0]
    recipe_mean_li.append(stats.describe(recipe_rating).mean)

print(stats.describe(recipe_mean_li))

"""
recipe_info_li = []
for i in range(total_depth):
    recipe_info = [encoded_recipe_title[i], recipe_categories[i], recipe_ingredients[i], recipe_directions[i]]
    recipe_info_li.append(recipe_info)

recipe_category_li = []
for i, recipe_category in enumerate(recipe_categories):
    try:
        lowered_category = ""
        for category in recipe_category:
            category = category.lower().replace(" ","_")
            if lowered_category == "":
                lowered_category = category
            else:
                lowered_category = lowered_category + ", " + category
        recipe_category_li.append(lowered_category)

    except ValueError:
        recipe_category_li.append('')

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=3, stop_words='english')
X = vectorizer.fit_transform(recipe_category_li[:100])

feature_names = vectorizer.get_feature_names()
print(feature_names)

from sklearn.metrics.pairwise import cosine_similarity
recipe_sim = cosine_similarity(X)
def similar_recommend_by_movie_id(recipe_category_li, recipes_id):
    recipe_index = recipes_id - 1
    similar_recipes = sorted(list(enumerate(recipe_sim[recipe_index])),key=lambda x:x[1], reverse=True)
    recommended = 1
    index = recipe_info_li[recipes_id]
    print("-----recommendation for recipe : %s -------"%(recipe_label_encoder.inverse_transform(index[0])))
    for recipe_info in similar_recipes[1:6]:
        # 주어진 영화와 가장 비슷한 영화는 그 영화 자신이므로 출력 시 제외합니다.
        recipe_title = recipe_info_li[recipe_info[0]]
        print('rank %d recommendation : %s'%(recommended, recipe_label_encoder.inverse_transform(recipe_title[0])))
        recommended+=1

similar_recommend_by_movie_id(recipe_category_li, 0)
"""

"""
from sklearn.metrics import mean_squared_error
import numpy as np
def compute_ALS(R, n_iter, lambda_, k):
    '''임의의 사용자 요인 행렬 X와 임의의 영화 요인 행렬 Y를 생성한 뒤
    교대 최소제곱법을 이용하여 유틸리티 행렬 R을 근사합니다.
    R(ndarray) : 유틸리티 행렬
    lambda_(float) : 정규화 파라미터입니다.
    n_iter(fint) : X와 Y의 갱신 횟수입니다.
    '''
    m, n =R.shape
    X = np.random.rand(m, k)
    Y = np.random.rand(k, n)
                            
    # 각 갱신 때마다 계산한 에러를 저장합니다.
    errors =[]
    for i in range(0, n_iter):
        # [식 6-4]를 구현했습니다.
        # 넘파이의 eye 함수는 파라미터 a를 받아 a x a 크기의 단위행렬을 만듭니다.
        X = np.linalg.solve(np.dot(Y, Y.T) + lambda_ * np.eye(k), np.dot(Y, R.T)).T
        Y = np.linalg.solve(np.dot(X.T, X) + lambda_ * np.eye(k), np.dot(X.T, R))
    
        errors.append(mean_squared_error(R, np.dot(X, Y)))

        if i % 10 == 0:
            print('iteration %d is completed'%(i))
    
    R_hat = np.dot(X, Y)
    print('Error of rated movies: %.5f'%(mean_squared_error(R, np.dot(X, Y))))
    return(R_hat, errors)

R_hat, errors = compute_ALS(R, 20, 0.1,100)
print(R_hat, errors)
"""
