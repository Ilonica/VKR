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
# MOVIELENS_DATA_SIZE = '20m'
# EPOCHS = 100
# BATCH_SIZE = 256
# SEED = DEFAULT_SEED
#
# # Загрузка данных
# df = movielens.load_pandas_df(
#     size=MOVIELENS_DATA_SIZE,
#     header=["userID", "itemID", "rating", "timestamp"]
# )
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
#     n_factors=4,
#     layer_sizes=[16, 8, 4],
#     n_epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     learning_rate=1e-3,
#     verbose=10,
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

import torch
print(torch.__version__)          # Должно быть с суффиксом cuXXX (например, cu113)
print(torch.cuda.is_available())  # Должно вернуть True