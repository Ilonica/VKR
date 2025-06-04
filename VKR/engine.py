import torch
import os
import glob
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import save_checkpoint, use_optimizer
from metrics import MetronAtK


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()

    def train_single_batch(self, users, items, ratings):
        assert hasattr(self, 'model'), 'Пожалуйста, укажите точную модель!'
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
        self.opt.zero_grad()
        ratings_pred = self.model(users, items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        loss.backward()
        self.opt.step()
        loss = loss.item()
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        assert hasattr(self, 'model'), 'Пожалуйста, укажите точную модель!'
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rating = batch[0], batch[1], batch[2]
            rating = rating.float()
            loss = self.train_single_batch(user, item, rating)
            print('[Эпоха обучения {}] Батч {}, Функция потерь {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)

    def evaluate(self, evaluate_data, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        self.model.eval()
        with torch.no_grad():
            test_users, test_items = evaluate_data[0], evaluate_data[1]
            negative_users, negative_items = evaluate_data[2], evaluate_data[3]

            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                negative_users = negative_users.cuda()
                negative_items = negative_items.cuda()

            test_scores = []
            negative_scores = []

            # Обработка батчей для тестовых данных
            bs = self.config['batch_size'] if self.config['use_bachify_eval'] else len(test_users)
            for start_idx in range(0, len(test_users), bs):
                end_idx = min(start_idx + bs, len(test_users))
                batch_test_users = test_users[start_idx:end_idx]
                batch_test_items = test_items[start_idx:end_idx]
                test_scores.append(self.model(batch_test_users, batch_test_items))

            # Обработка батчей для негативных данных
            for start_idx in range(0, len(negative_users), bs):
                end_idx = min(start_idx + bs, len(negative_users))
                batch_negative_users = negative_users[start_idx:end_idx]
                batch_negative_items = negative_items[start_idx:end_idx]
                negative_scores.append(self.model(batch_negative_users, batch_negative_items))

            # Объединение результатов
            test_scores = torch.cat(test_scores, dim=0)
            negative_scores = torch.cat(negative_scores, dim=0)

            # Перенос данных на CPU, если используется CUDA
            if self.config['use_cuda'] is True:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_scores = test_scores.cpu()
                negative_users = negative_users.cpu()
                negative_items = negative_items.cpu()
                negative_scores = negative_scores.cpu()

            # Установка subjects для MetronAtK
            self._metron.subjects = [
                test_users.data.view(-1).tolist(),
                test_items.data.view(-1).tolist(),
                test_scores.data.view(-1).tolist(),
                negative_users.data.view(-1).tolist(),
                negative_items.data.view(-1).tolist(),
                negative_scores.data.view(-1).tolist()
            ]

        # Расчет метрик
        hit_ratio = self._metron.cal_hit_ratio()
        ndcg = self._metron.cal_ndcg()

        # Логирование в TensorBoard
        self._writer.add_scalar('performance/HR', hit_ratio, epoch_id)
        self._writer.add_scalar('performance/NDCG', ndcg, epoch_id)
        print('[Evaluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}'.format(epoch_id, hit_ratio, ndcg))
        return hit_ratio, ndcg

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        """Сохраняет полное состояние обучения"""
        assert hasattr(self, 'model'), 'Please specify the exact model!'

        checkpoint = {
            'epoch': epoch_id,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'hr': hit_ratio,
            'ndcg': ndcg,
            'config': self.config
        }

        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        torch.save(checkpoint, model_dir)
        return model_dir

    def convert_old_checkpoint(self, old_checkpoint_path, new_checkpoint_path):
        """Конвертирует старый чекпоинт в новый формат"""
        old_state = torch.load(old_checkpoint_path)
        new_state = {
            'epoch': 0,  # Неизвестно для старых чекпоинтов
            'model_state_dict': old_state,
            'optimizer_state_dict': self.opt.state_dict(),  # Инициализирует новый
            'hr': 0.0,
            'ndcg': 0.0,
            'config': self.config
        }
        torch.save(new_state, new_checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """Загружает полное состояние обучения"""
        if os.path.exists(checkpoint_path):
            # Добавляем map_location для совместимости CPU/GPU
            map_location = 'cuda' if self.config['use_cuda'] else 'cpu'
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

            # Проверяем формат чекпоинта
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
                return checkpoint['epoch'], checkpoint['hr'], checkpoint['ndcg']
            else:
                # Совместимость со старым форматом (только веса модели)
                self.model.load_state_dict(checkpoint)
                return 0, 0.0, 0.0
        return 0, 0.0, 0.0

    def find_last_checkpoint(self, alias):
        """Поиск последнего чекпоинта"""
        checkpoint_files = glob.glob(f'checkpoints/{alias}_Epoch*')
        if not checkpoint_files:
            return None

        # Извлекаем номера эпох и находим максимальный
        epochs = [int(f.split('_Epoch')[1].split('_')[0]) for f in checkpoint_files]
        last_idx = epochs.index(max(epochs))
        return checkpoint_files[last_idx]

