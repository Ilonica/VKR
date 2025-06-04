import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from ode import CDECFEngine
from neumf import NeuMFEngine
from data import SampleGenerator

def log_metrics(epoch, hr, ndcg, filename='metrics_log.txt'):
    with open(filename, 'a') as f:
        f.write(f'Epoch {epoch}: HR = {hr:.4f}, NDCG = {ndcg:.4f}\n')

gmf_config = {'alias': 'gmf_factor8neg4-implict',
              'num_epoch': 200,
              'batch_size': 1024,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0,  # 0.01
              'weight_init_gaussian': True,
              'use_cuda': False,
              'use_bachify_eval': False,
              'device_id': 0,
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 5,
              'batch_size': 1024,  # 256, 1024
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 6040,
              'num_items': 3706,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'weight_init_gaussian': True,
              'use_cuda': False,
              'use_bachify_eval': False,
              'device_id': 0,
              'pretrain': False, # Важно: без предобучения
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'neumf_factor8neg4',
                'num_epoch': 200,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 6040,
                'num_items': 3706,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16, 64, 32, 16, 8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.0000001,
                'weight_init_gaussian': True,
                'use_cuda': False,
                'use_bachify_eval': True,
                'device_id': 0,
                'pretrain': False,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

ode_config = {
    'alias': 'cdecf_factor8neg4',  # Пример имени
    'num_epoch': 200,
    'batch_size': 1024,
    'optimizer': 'adam',
    'adam_lr': 1e-3,
    'num_users': 6040,
    'num_items': 3706,
    'latent_dim': 8,
    'num_negative': 4,
    'layers': [16, 64, 32, 16, 8],
    'l2_regularization': 0.0000001,
    'weight_init_gaussian': True,
    'use_cuda': True,
    'use_bachify_eval': True,
    'device_id': 0,
    'pretrain': False,
    'pretrain_mf': 'checkpoints/gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model',  # Путь к GMF
    'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
}

cde_cf_config = {
    'alias': 'cdecf_factor8neg4',
    'num_epoch': 200,
    'batch_size': 1024,
    'optimizer': 'adam',
    'adam_lr': 1e-3,
    'num_users': 6040,
    'num_items': 3706,
    'latent_dim': 8,
    'num_negative': 4,
    'layers': [16, 64, 32, 16, 8],
    'l2_regularization': 0.0000001,  # weight_decay
    'weight_init_gaussian': True,
    'use_cuda': False,
    'use_bachify_eval': True,
    'device_id': 0,
    'ode_time': 1.0,  # Общее время интегрирования ODE
    'ode_steps': 10,  # Количество шагов интегрирования
    'ode_solver': 'dopri5',  # Метод решения ODE ('euler', 'rk4', 'dopri5' и т.д.)
    'weight_decay': 1e-5,  # Коэффициент L2 регуляризации
    'pretrain': False,
    'pretrain_mf': 'checkpoints/gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model',
    'model_dir': 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
}

# Load Data
ml1m_dir = 'data/ml-1m/ratings.dat'
ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'], engine='python')
# Reindex
user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
item_id = ml1m_rating[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
# DataLoader for training
sample_generator = SampleGenerator(ratings=ml1m_rating)
evaluate_data = sample_generator.evaluate_data
# Specify the exact model
# config = gmf_config
# engine = GMFEngine(config)
# config = mlp_config
# engine = MLPEngine(config)
# config = ode_config
config = cde_cf_config
engine = CDECFEngine(config)
# config = neumf_config
# engine = NeuMFEngine(config)

# Загрузка последнего чекпоинта
last_checkpoint = engine.find_last_checkpoint(config['alias'])
if last_checkpoint:
    print(f"Найден чекпоинт: {last_checkpoint}")
    try:
        start_epoch, last_hr, last_ndcg = engine.load_checkpoint(last_checkpoint)
        start_epoch += 1  # Продолжаем со следующей эпохи
        print(f"Продолжаем обучение с эпохи {start_epoch}, последние метрики: HR={last_hr:.4f}, NDCG={last_ndcg:.4f}")
    except Exception as e:
        print(f"Ошибка загрузки чекпоинта: {str(e)}")
        print("Начинаем обучение с нуля")
        start_epoch = 0
else:
    start_epoch = 0
    print("Чекпоинты не найдены, начинаем обучение с нуля")

# Очистка файла метрик только при начале нового обучения
if start_epoch == 0:
    open('metrics_log.txt', 'w').close()

for epoch in range(start_epoch, config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config['alias'], epoch, hit_ratio, ndcg)

    # Запись метрики в файл
    log_metrics(epoch, hit_ratio, ndcg)
