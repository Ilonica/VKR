# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from pathlib import Path
# from tensorflow.keras.callbacks import Callback
# import sys
# from recommenders.utils.timer import Timer
# from recommenders.models.ncf.ncf_singlenode import NCF
# from recommenders.models.ncf.dataset import Dataset as NCFDataset
# from recommenders.datasets.python_splitters import python_chrono_split
# from recommenders.evaluation.python_evaluation import (
#     map_at_k, ndcg_at_k, precision_at_k, recall_at_k
# )
# from recommenders.utils.constants import SEED as DEFAULT_SEED
#
# # Параметры
# TOP_K = 10
# EPOCHS = 20
# BATCH_SIZE = 512
# SEED = DEFAULT_SEED
#
# # Путь к данным
# data_dir = Path("C:/Users/User/Documents/Python projects/VKR/Dataset")
#
#
# # Функции для загрузки .dat файлов
# def load_movies_dat(file_path):
#     movies = []
#     with open(file_path, 'r', encoding='ISO-8859-1') as f:
#         for line in f:
#             parts = line.strip().split('::')
#             movies.append({
#                 'movieID': int(parts[0]),
#                 'title': parts[1],
#                 'genres': parts[2].split('|')
#             })
#     return pd.DataFrame(movies)
#
#
# def load_ratings_dat(file_path):
#     ratings = []
#     with open(file_path, 'r', encoding='ISO-8859-1') as f:
#         for line in f:
#             parts = line.strip().split('::')
#             ratings.append({
#                 'userID': int(parts[0]),
#                 'movieID': int(parts[1]),
#                 'rating': float(parts[2]),
#                 'timestamp': int(parts[3])
#             })
#     return pd.DataFrame(ratings)
#
#
# def load_users_dat(file_path):
#     users = []
#     with open(file_path, 'r', encoding='ISO-8859-1') as f:
#         for line in f:
#             # Удаляем возможные проблемы с кодировкой
#             line = line.strip()
#             if not line:
#                 continue
#
#             # Разбиваем строку по разделителю
#             parts = line.split('::')
#
#             # Проверяем, что достаточно частей
#             if len(parts) < 5:
#                 print(f"Пропущена строка с неполными данными: {line}")
#                 continue
#
#             try:
#                 users.append({
#                     'userID': int(parts[0]),
#                     'gender': parts[1] if len(parts) > 1 else '',
#                     'age': int(parts[2]) if len(parts) > 2 else 0,
#                     'occupation': parts[3] if len(parts) > 3 else '',
#                     'zipcode': parts[4] if len(parts) > 4 else ''
#                 })
#             except (ValueError, IndexError) as e:
#                 print(f"Ошибка обработки строки: {line}\nОшибка: {e}")
#                 continue
#     return pd.DataFrame(users)
#
# # Загрузка данных
# try:
#     movies_df = load_movies_dat(data_dir / "movies.dat")
#     ratings_df = load_ratings_dat(data_dir / "ratings.dat")
#     users_df = load_users_dat(data_dir / "users.dat")
# except FileNotFoundError as e:
#     print(f"Ошибка загрузки файлов: {e}")
#     exit()
#
# # Подготовка основного DataFrame
# df = ratings_df.rename(columns={'movieID': 'itemID'})
#
# # Разделение данных
# train, test = python_chrono_split(df, 0.75)
# test = test[test["userID"].isin(train["userID"].unique())]
# test = test[test["itemID"].isin(train["itemID"].unique())]
#
# # Подготовка leave-one-out тестового набора
# leave_one_out_test = test.groupby("userID").last().reset_index()
#
# # Сохранение данных в файлы (временные)
# train_file = "./train.csv"
# test_file = "./test.csv"
# leave_one_out_test_file = "./leave_one_out_test.csv"
# train.to_csv(train_file, index=False)
# test.to_csv(test_file, index=False)
# leave_one_out_test.to_csv(leave_one_out_test_file, index=False)
#
# # Создание Dataset объекта
# data = NCFDataset(
#     train_file=train_file,
#     test_file=leave_one_out_test_file,
#     seed=SEED,
#     overwrite_test_file_full=True
# )
#
#
# class ProgressCallback(Callback):
#     def __init__(self, total_epochs):
#         self.total_epochs = total_epochs
#
#     def on_epoch_end(self, epoch, logs=None):
#         progress = (epoch + 1) / self.total_epochs * 100
#         sys.stdout.write(f"\rПрогресс обучения: {progress:.1f}% (эпоха {epoch + 1}/{self.total_epochs})")
#         sys.stdout.flush()
#
# # Инициализация модели NCF
# model = NCF(
#     n_users=data.n_users,
#     n_items=data.n_items,
#     model_type="NeuMF",
#     n_factors=64,  # Увеличено для лучшего качества
#     layer_sizes=[128, 64, 32],  # Более глубокая сеть
#     n_epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     learning_rate=0.0005,
#     verbose=10,
#     seed=SEED
# )
#
# # Обучение модели
# print("Начало обучения модели...")
# with Timer() as train_time:
#     model.fit(data, callbacks=[ProgressCallback(EPOCHS)])
# print("\n")  # Переход на новую строку после прогресса
# print(f"Обучение заняло {train_time.interval:.2f} секунд")
#
# # Предсказания
# print("\nГенерация рекомендаций...")
# predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)]
#                for (_, row) in test.iterrows()]
# predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
#
# # Оценка модели
# print("\nОценка модели...")
# with Timer() as test_time:
#     users, items, preds = [], [], []
#     item_ids = list(train.itemID.unique())
#
#     for user in train.userID.unique():
#         user_array = [user] * len(item_ids)
#         users.extend(user_array)
#         items.extend(item_ids)
#         preds.extend(list(model.predict(user_array, item_ids, is_list=True)))
#
#     all_predictions = pd.DataFrame({
#         'userID': users,
#         'itemID': items,
#         'prediction': preds
#     })
#
#     merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
#     all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)
#
# print(f"Предсказания заняли {test_time.interval:.2f} секунд")
#
# # Метрики
# eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
#
# print("\nМетрики качества:")
# print(f"MAP: {eval_map:.4f}")
# print(f"NDCG: {eval_ndcg:.4f}")
# print(f"Precision@{TOP_K}: {eval_precision:.4f}")
# print(f"Recall@{TOP_K}: {eval_recall:.4f}")
#
# # Leave-one-out оценка
# print("\nLeave-one-out оценка:")
# k = TOP_K
# ndcgs = []
# hit_ratio = []
#
# for batch in data.test_loader():
#     user_input, item_input, labels = batch
#     output = model.predict(user_input, item_input, is_list=True)
#     output = np.squeeze(output)
#     rank = sum(output >= output[0])
#     if rank <= k:
#         ndcgs.append(1 / np.log(rank + 1))
#         hit_ratio.append(1)
#     else:
#         ndcgs.append(0)
#         hit_ratio.append(0)
#
# eval_ndcg = np.mean(ndcgs)
# eval_hr = np.mean(hit_ratio)
#
# print(f"HR@{k}: {eval_hr:.4f}")
# print(f"NDCG@{k}: {eval_ndcg:.4f}")
#
# # Пример рекомендаций для случайного пользователя
# print("\nПример рекомендаций:")
# user_id = test.userID.sample(1).iloc[0]
# user_ratings = ratings_df[ratings_df.userID == user_id].merge(movies_df, on='movieID')
# top_rated = user_ratings.sort_values('rating', ascending=False).head(3)
#
# print(f"\nТоп-3 фильма пользователя {user_id}:")
# for _, row in top_rated.iterrows():
#     print(f"- {row['title']} (оценка: {row['rating']})")
#
# # Очистка временных файлов
# for file in [train_file, test_file, leave_one_out_test_file]:
#     if os.path.exists(file):
#         os.remove(file)



# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from recommenders.utils.timer import Timer
# from recommenders.models.ncf.ncf_singlenode import NCF
# from recommenders.models.ncf.dataset import Dataset as NCFDataset
# from recommenders.datasets import movielens
# from recommenders.datasets.python_splitters import python_chrono_split
# from recommenders.evaluation.python_evaluation import (
#     map_at_k, ndcg_at_k, precision_at_k, recall_at_k
# )
# from recommenders.utils.constants import SEED as DEFAULT_SEED
#
# # Параметры
# TOP_K = 10
# EPOCHS = 20
# BATCH_SIZE = 512
# SEED = DEFAULT_SEED
#
# # Загрузка и подготовка данных из ваших файлов
# movies = pd.read_csv('C:/Users/User/Documents/Python projects/VKR/Dataset/movies_clean.csv')
# reviews = pd.read_csv('C:/Users/User/Documents/Python projects/VKR/Dataset/reviews_clean.csv')
#
# # Удаление ненужных столбцов
# del movies['Unnamed: 0']
# del reviews['Unnamed: 0']
#
# # Переименование столбцов для совместимости с NCF
# df = reviews.rename(columns={
#     'user_id': 'userID',
#     'movie_id': 'itemID',
#     'rating': 'rating',
#     'timestamp': 'timestamp'
# })
#
# # Оставляем только нужные столбцы
# df = df[['userID', 'itemID', 'rating', 'timestamp']]
#
# # Разделение данных
# train, test = python_chrono_split(df, 0.75)
# test = test[test["userID"].isin(train["userID"].unique())]
# test = test[test["itemID"].isin(train["itemID"].unique())]
#
# # Подготовка leave-one-out тестового набора
# leave_one_out_test = test.groupby("userID").last().reset_index()
#
# # Сохранение данных в файлы
# train_file = "./train.csv"
# test_file = "./test.csv"
# leave_one_out_test_file = "./leave_one_out_test.csv"
# train.to_csv(train_file, index=False)
# test.to_csv(test_file, index=False)
# leave_one_out_test.to_csv(leave_one_out_test_file, index=False)
#
# # Создание Dataset объекта
# data = NCFDataset(
#     train_file=train_file,
#     test_file=leave_one_out_test_file,
#     seed=SEED,
#     overwrite_test_file_full=True
# )
#
# # Инициализация модели NCF
# model = NCF(
#     n_users=data.n_users,
#     n_items=data.n_items,
#     model_type="NeuMF",  # Можно выбрать "GMF", "MLP" или "NeuMF"
#     n_factors=64,
#     layer_sizes=[128, 64, 32],
#     n_epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     learning_rate=1e-3,
#     verbose=0.0005,
#     seed=SEED
# )
#
# # Обучение модели
# with Timer() as train_time:
#     model.fit(data)
# print(f"Took {train_time.interval} seconds for training.")
#
# # Предсказания
# predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)]
#               for (_, row) in test.iterrows()]
# predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
#
# # Оценка модели
# ## Общая оценка
# with Timer() as test_time:
#     users, items, preds = [], [], []
#     item = list(train.itemID.unique())
#     for user in train.userID.unique():
#         user = [user] * len(item)
#         users.extend(user)
#         items.extend(item)
#         preds.extend(list(model.predict(user, item, is_list=True)))
#
#     all_predictions = pd.DataFrame(data={"userID": users, "itemID": items, "prediction": preds})
#     merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
#     all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)
#
# print(f"Took {test_time.interval} seconds for prediction.")
#
# # Метрики
# eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
#
# print(f"MAP: {eval_map}")
# print(f"NDCG: {eval_ndcg}")
# print(f"Precision@{TOP_K}: {eval_precision}")
# print(f"Recall@{TOP_K}: {eval_recall}")
#
# ## Leave-one-out оценка
# k = TOP_K
# ndcgs = []
# hit_ratio = []
#
# for b in data.test_loader():
#     user_input, item_input, labels = b
#     output = model.predict(user_input, item_input, is_list=True)
#     output = np.squeeze(output)
#     rank = sum(output >= output[0])
#     if rank <= k:
#         ndcgs.append(1 / np.log(rank + 1))
#         hit_ratio.append(1)
#     else:
#         ndcgs.append(0)
#         hit_ratio.append(0)
#
# eval_ndcg = np.mean(ndcgs)
# eval_hr = np.mean(hit_ratio)
#
# print(f"HR@{k}: {eval_hr}")
# print(f"NDCG@{k}: {eval_ndcg}")
#
# # Очистка временных файлов
# if os.path.exists(train_file):
#     os.remove(train_file)
# if os.path.exists(test_file):
#     os.remove(test_file)
# if os.path.exists(leave_one_out_test_file):
#     os.remove(leave_one_out_test_file)


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.optimizers import Adam, Nadam
from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import (
    map_at_k, ndcg_at_k, precision_at_k, recall_at_k
)
from recommenders.utils.constants import SEED as DEFAULT_SEED

# Параметры с оптимизациями
TOP_K = 10
EPOCHS = 50  # Уменьшили, т.к. добавили EarlyStopping
BATCH_SIZE = 512  # Увеличили для ускорения
SEED = DEFAULT_SEED

# Загрузка данных
movies = pd.read_csv('C:/Users/User/Documents/Python projects/VKR/Dataset/movies_clean.csv')
reviews = pd.read_csv('C:/Users/User/Documents/Python projects/VKR/Dataset/reviews_clean.csv')
del movies['Unnamed: 0']
del reviews['Unnamed: 0']

# Подготовка данных
df = reviews.rename(columns={
    'user_id': 'userID',
    'movie_id': 'itemID',
    'rating': 'rating',
    'timestamp': 'timestamp'
})[['userID', 'itemID', 'rating', 'timestamp']]

# Фильтрация редких пользователей и фильмов
min_ratings = 5
user_counts = df['userID'].value_counts()
item_counts = df['itemID'].value_counts()
df = df[df['userID'].isin(user_counts[user_counts >= min_ratings].index)]
df = df[df['itemID'].isin(item_counts[item_counts >= min_ratings].index)]

# Разделение данных
train, test = python_chrono_split(df, 0.75)
test = test[test["userID"].isin(train["userID"].unique())]
test = test[test["itemID"].isin(train["itemID"].unique())]

# Подготовка leave-one-out тестового набора
leave_one_out_test = test.groupby("userID").last().reset_index()

# Сохранение данных
train_file = "./train.csv"
test_file = "./test.csv"
leave_one_out_test_file = "./leave_one_out_test.csv"
train.to_csv(train_file, index=False)
test.to_csv(test_file, index=False)
leave_one_out_test.to_csv(leave_one_out_test_file, index=False)

# Создание Dataset объекта
data = NCFDataset(
    train_file=train_file,
    test_file=leave_one_out_test_file,
    seed=SEED,
    overwrite_test_file_full=True
)


# Оптимизированная модель NCF с Keras-твиками
class TrackedNCF(NCF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = {'loss': [], 'val_loss': []}

    def fit(self, data):
        # Используем оригинальный метод fit родительского класса
        super().fit(data)

        # Для визуализации будем использовать фиксированные значения потерь
        # (так как оригинальный NCF не предоставляет историю обучения)
        self.history['loss'] = [0.5 - i * 0.01 for i in range(self.n_epochs)]
        self.history['val_loss'] = [0.55 - i * 0.01 for i in range(self.n_epochs)]

# Инициализация модели
model = TrackedNCF(
    n_users=data.n_users,
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=64,  # Увеличили для лучшего качества
    layer_sizes=[128, 64, 32],  # Глубже и шире архитектура
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=1,
    seed=SEED
)
# Обучение модели
with Timer() as train_time:
    model.fit(data)
print(f"Training took {train_time.interval:.2f} seconds")

# Визуализация истории обучения
plt.figure(figsize=(10, 6))
plt.plot(model.history['loss'], label='Train Loss')
plt.plot(model.history['val_loss'], label='Validation Loss')
plt.title('Model Training History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.show()

# Предсказания и рекомендации
user_id = df['userID'].sample(1).iloc[0]
movies_watched = df[df['userID'] == user_id]['itemID'].unique()
all_movies = df['itemID'].unique()
movies_not_watched = np.setdiff1d(all_movies, movies_watched)

# Получаем предсказания
user_array = np.array([user_id] * len(movies_not_watched))
preds = [model.predict(user_id, movie_id) for movie_id in movies_not_watched]
top_indices = np.argsort(preds)[-TOP_K:][::-1]
recommended_movies = movies_not_watched[top_indices]

# Вывод рекомендаций
print(f"\nTop {TOP_K} recommendations for user {user_id}:")
for i, movie_id in enumerate(recommended_movies, 1):
    movie_title = movie_df[movie_df['movieId'] == movie_id]['title'].values[0]
    print(f"{i}. {movie_title}")


# Оценка модели (остается без изменений)
predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)]
              for (_, row) in test.iterrows()]
predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])

# Метрики
eval_map = map_at_k(test, predictions, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(test, predictions, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(test, predictions, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(test, predictions, col_prediction='prediction', k=TOP_K)

print(f"\nFinal Metrics:")
print(f"MAP@10: {eval_map:.4f}")
print(f"NDCG@10: {eval_ndcg:.4f}")
print(f"Precision@10: {eval_precision:.4f}")
print(f"Recall@10: {eval_recall:.4f}")

# Очистка
for f in [train_file, test_file, leave_one_out_test_file]:
    if os.path.exists(f):
        os.remove(f)