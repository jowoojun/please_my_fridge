# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pymongo import MongoClient
import os
from sklearn.metrics import mean_squared_error
import tensorflow as tf

client = MongoClient('ds119049.mlab.com', 19049)

print("connecting collections")
db = client['please_my_fridge']
db.authenticate(os.environ["mongoID"], os.environ["mongoPW"])
recipes = db.full_recipes
users = db.full_user_data
#recipes = db.fake_recipe_data
#users = db.fake_user_data

# data preprocessing
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


# encode recipe and users_nickname
from sklearn import preprocessing
recipe_label_encoder = preprocessing.LabelEncoder()
recipe_label_encoder.fit(recipe_title)
encoded_recipe_title = np.array(recipe_label_encoder.transform(recipe_title), dtype=np.int32)
total_recipe_depth = encoded_recipe_title.max() + 1

nickname_label_encoder = preprocessing.LabelEncoder()
nickname_label_encoder.fit(user_nick_name)
encoded_user_nick_name = np.array(nickname_label_encoder.transform(user_nick_name), dtype=np.int32)
total_user_depth = encoded_user_nick_name.max() + 1


# make user-recipe-matrix
def read_rating_data():
    Q = np.zeros((total_user_depth, total_recipe_depth), dtype=np.float64)

    for i, nick in enumerate(encoded_user_nick_name):
        row = nick
        for j, tr in enumerate(user_tried[nick]):
            col = recipe_label_encoder.transform([tr])
            Q[row, col[0]] = float(1.0)

    return Q

R = read_rating_data()


# Convert DataFrame in user-item matrix
matrix = pd.DataFrame(R, columns=[i for i in range(total_recipe_depth)], index=[j for j in range(total_user_depth)])

users = matrix.index.tolist()
recipes = matrix.columns.tolist()
matrix = matrix.as_matrix()


# tensorflow declare
input_size = total_recipe_depth
hidden_layer_size_1 = 10
hidden_layer_size_2 = 5

X = tf.placeholder(tf.float64, [None, input_size])

weights = {
    'encoder_w1': tf.Variable(tf.random_normal([input_size, hidden_layer_size_1], dtype=tf.float64)),
    'encoder_w2': tf.Variable(tf.random_normal([hidden_layer_size_1, hidden_layer_size_2], dtype=tf.float64)),
    'decoder_w1': tf.Variable(tf.random_normal([hidden_layer_size_2, hidden_layer_size_1], dtype=tf.float64)),
    'decoder_w2': tf.Variable(tf.random_normal([hidden_layer_size_1, input_size], dtype=tf.float64)),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([hidden_layer_size_1], dtype=tf.float64)),
    'encoder_b2': tf.Variable(tf.random_normal([hidden_layer_size_2], dtype=tf.float64)),
    'decoder_b1': tf.Variable(tf.random_normal([hidden_layer_size_1], dtype=tf.float64)),
    'decoder_b2': tf.Variable(tf.random_normal([input_size], dtype=tf.float64)),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_w1']), biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_w2']), biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_w1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_w2']), biases['decoder_b2']))
    return layer_2


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op

# Targets are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.losses.mean_squared_error(y_true, y_pred)
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)

predictions = pd.DataFrame()

# Define evaluation metrics
eval_x = tf.placeholder(tf.int32)
eval_y = tf.placeholder(tf.int32)
pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()

# Session
with tf.Session() as session:
    epochs = 100
    batch_size = 2

    session.run(init)
    session.run(local_init)

    num_batches = int(matrix.shape[0] / batch_size)
    matrix = np.array_split(matrix, num_batches)

    for i in range(epochs):

        avg_cost = 0

        for batch in matrix:
            _, l = session.run([optimizer, loss], feed_dict={X: batch})
            avg_cost += l

        avg_cost /= num_batches

        print("Epoch: {} Loss: {}".format(i + 1, avg_cost))

    print("Predictions...")

    matrix = np.concatenate(matrix, axis=0)
    preds = session.run(decoder_op, feed_dict={X: matrix})
    predictions = predictions.append(pd.DataFrame(preds))

    predictions = predictions.stack().reset_index(name='rating')
    predictions.columns = ['users', 'recipe', 'rating']
    predictions['users'] = predictions['users'].map(lambda value: users[value])
    predictions['recipe'] = predictions['recipe'].map(lambda value: recipes[value])

    print("Filtering out recipes in training set")
    keys = ['users', 'recipe']

    df = pd.DataFrame(R, columns=[i for i in range(total_recipe_depth)], index=[j for j in range(total_user_depth)])
    df = df.stack().reset_index(name='rating')
    df.columns = ['users', 'recipe', 'rating']
    df['users'] = df['users'].map(lambda value: users[value])
    df['recipe'] = df['recipe'].map(lambda value: recipes[value])

    i1 = predictions.set_index(keys).index
    i2 = df.set_index(keys).index

    """
    recs = predictions[~i1.isin(i2)]
    recs = recs.sort_values(['users', 'rating'], ascending=[True, False])
    recs = recs.groupby('users').head(1)
    recs.to_csv('recs.tsv', sep='\t', index=False, header=False)
    """

    # test
    print("Test: [1,0,0,0,0]")
    print("If you tried only 'a' food, we recommend below foods")
    #tried = ["a", "e"]
    tried = ["Lentil Apple and Turkey Wrap","Maple Mustard Sauce", "Potato and Fennel Soup Hodge","Mahi-Mahi in Tomato Olive Sauce"]
    encoded_tried = np.zeros((total_recipe_depth))
    col = recipe_label_encoder.transform(tried)
    for co in col:
        encoded_tried[co] = 1

    encoded_tried = np.expand_dims(encoded_tried, axis=0)
    recommand = session.run(decoder_op, feed_dict={X: encoded_tried})
    print(recommand)

    arg_recommand = np.argsort(recommand)[0][::-1]
    for arg in arg_recommand[0:10]:
        print(arg, recipe_label_encoder.inverse_transform(arg))

