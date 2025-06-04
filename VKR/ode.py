import torch
from torch import nn
from torchdiffeq import odeint
from gmf import GMF
from engine import Engine
from utils import use_cuda, resume_checkpoint
from torchdiffeq import odeint


# class ODEFunc(nn.Module):
#     def __init__(self, latent_dim, layers):
#         super().__init__()
#         self.mlp_controller = nn.Sequential(
#             nn.Linear(latent_dim * 2, 64),
#             nn.ReLU(),
#             nn.Linear(64, latent_dim),
#             nn.Sigmoid()
#         )
#         self.gcn_transform = nn.Linear(latent_dim, latent_dim)
#
#     def forward(self, t, x):
#         # x: [batch_size, latent_dim * 2] (concat user_emb + item_emb)
#         user_emb, item_emb = x.chunk(2, dim=1)
#
#         # Генерация весов
#         weights = self.mlp_controller(x)
#
#         # Графовая трансформация (упрощённый аналог GCN)
#         transformed = self.gcn_transform(item_emb) * weights
#         return torch.cat([transformed, torch.zeros_like(transformed)], dim=1)
#
#
# class CDECF(torch.nn.Module):
#     def __init__(self, config):
#         super(CDECF, self).__init__()
#         self.config = config
#         self.num_users = config['num_users']
#         self.num_items = config['num_items']
#         self.latent_dim = config['latent_dim']
#
#         # Эмбеддинги как в оригинальном MLP
#         self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
#         self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
#
#         # Neural ODE компонент (замена FC слоям MLP)
#         self.ode_func = ODEFunc(self.latent_dim, config['layers'])
#         self.time_steps = torch.linspace(0, 1, steps=6)
#
#         # Финальные слои
#         self.affine_output = nn.Linear(self.latent_dim * 2, 1)  # concat(user_emb, item_emb)
#         self.logistic = nn.Sigmoid()
#
#         # Инициализация весов
#         if config['weight_init_gaussian']:
#             for m in self.modules():
#                 if isinstance(m, (nn.Embedding, nn.Linear)):
#                     nn.init.normal_(m.weight, 0, 0.01)
#
#     def forward(self, user_indices, item_indices):
#         user_emb = self.embedding_user(user_indices)
#         item_emb = self.embedding_item(item_indices)
#
#         # Объединение эмбеддингов для ODE
#         ode_input = torch.cat([user_emb, item_emb], dim=1)
#
#         # ODE
#         ode_output = odeint(
#             self.ode_func,
#             ode_input,
#             self.time_steps,
#             method='dopri5'  # Метод RK4
#         )[-1]
#
#         # Разделяем обновлённые эмбеддинги
#         user_emb_updated, _ = ode_output.chunk(2, dim=1)
#
#         # Конкатенация и предсказание
#         vector = torch.cat([user_emb_updated, item_emb], dim=1)
#         logits = self.affine_output(vector)
#         return self.logistic(logits)
#
#     # Загрузка весов из GMF
#     def load_pretrain_weights(self):
#         gmf_model = GMF(self.config)
#         if self.config['use_cuda']:
#             gmf_model.cuda()
#         resume_checkpoint(gmf_model, self.config['pretrain_mf'], self.config['device_id'])
#         self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
#         self.embedding_item.weight.data = gmf_model.embedding_item.weight.data
#
#
# class CDECFEngine(Engine):
#     def __init__(self, config):
#         self.model = CDECF(config)
#         if config['use_cuda']:
#             use_cuda(True, config['device_id'])
#             self.model.cuda()
#         super(CDECFEngine, self).__init__(config)
#
#         if config['pretrain']:
#             self.model.load_pretrain_weights()


# class CDEFunc(nn.Module):
#     def __init__(self, latent_dim, layers):
#         super().__init__()
#         # Weight generator MLP (controller)
#         self.mlp_controller = nn.Sequential(
#             nn.Linear(latent_dim * 2, 64),
#             nn.ReLU(),
#             nn.Linear(64, latent_dim),
#             nn.Sigmoid()
#         )
#         # Graph transformation (simplified GCN)
#         self.gcn_transform = nn.Linear(latent_dim, latent_dim)
#
#     def forward(self, t, x):
#         # x: [batch_size, latent_dim * 2] (concat user_emb + item_emb)
#         user_emb, item_emb = x.chunk(2, dim=1)
#
#         # Generate continuous weights based on current embeddings and time
#         weights = self.mlp_controller(x)
#
#         # Graph transformation with controlled weights
#         user_transformed = self.gcn_transform(user_emb) * weights
#         item_transformed = self.gcn_transform(item_emb) * weights
#
#         # Return derivatives for both user and item embeddings
#         return torch.cat([user_transformed, item_transformed], dim=1)
#
#
# class CDECF(torch.nn.Module):
#     def __init__(self, config):
#         super(CDECF, self).__init__()
#         self.config = config
#         self.num_users = config['num_users']
#         self.num_items = config['num_items']
#         self.latent_dim = config['latent_dim']
#
#         # Embeddings
#         self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
#         self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
#
#         # CDE component (replaces ODE)
#         self.cde_func = CDEFunc(self.latent_dim, config['layers'])
#         self.time_steps = torch.linspace(0, 1, steps=6)  # Reduced from original ODE's 8.5 to ~6.5
#
#         # Final layers
#         self.affine_output = nn.Linear(self.latent_dim * 2, 1)
#         self.logistic = nn.Sigmoid()
#
#         # Weight initialization
#         if config['weight_init_gaussian']:
#             for m in self.modules():
#                 if isinstance(m, (nn.Embedding, nn.Linear)):
#                     nn.init.normal_(m.weight, 0, 0.01)
#
#     def forward(self, user_indices, item_indices):
#         user_emb = self.embedding_user(user_indices)
#         item_emb = self.embedding_item(item_indices)
#
#         # Combine embeddings for CDE
#         cde_input = torch.cat([user_emb, item_emb], dim=1)
#
#         # Solve the Controlled Differential Equation
#         cde_output = odeint(
#             self.cde_func,
#             cde_input,
#             self.time_steps,
#             method='dopri5'  # RK4 method
#         )[-1]  # Take the final state
#
#         # Split the updated embeddings
#         user_emb_updated, item_emb_updated = cde_output.chunk(2, dim=1)
#
#         # Concatenate and predict
#         vector = torch.cat([user_emb_updated, item_emb_updated], dim=1)
#         logits = self.affine_output(vector)
#         return self.logistic(logits)
#
#     def load_pretrain_weights(self):
#         gmf_model = GMF(self.config)
#         if self.config['use_cuda']:
#             gmf_model.cuda()
#         resume_checkpoint(gmf_model, self.config['pretrain_mf'], self.config['device_id'])
#         self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
#         self.embedding_item.weight.data = gmf_model.embedding_item.weight.data
#
#
# class CDECFEngine(Engine):
#     def __init__(self, config):
#         self.model = CDECF(config)
#         if config['use_cuda']:
#             use_cuda(True, config['device_id'])
#             self.model.cuda()
#         super(CDECFEngine, self).__init__(config)
#
#         if config['pretrain']:
#             self.model.load_pretrain_weights()




# class ODEFunc(nn.Module):
#     def __init__(self, latent_dim, layers):
#         super().__init__()
#         # Improved MLP controller with more layers for better representation
#         self.mlp_controller = nn.Sequential(
#             nn.Linear(latent_dim * 2, layers[1]),
#             nn.ReLU(),
#             nn.Linear(layers[1], layers[2]),
#             nn.ReLU(),
#             nn.Linear(layers[2], latent_dim),
#             nn.Sigmoid()
#         )
#         self.gcn_transform = nn.Linear(latent_dim, latent_dim)
#
#         # Initialize weights for better convergence
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, t, x):
#         # x: [batch_size, latent_dim * 2] (concat user_emb + item_emb)
#         user_emb, item_emb = x.chunk(2, dim=1)
#
#         # Генерация весов
#         weights = self.mlp_controller(x)
#
#         # Графовая трансформация (упрощённый аналог GCN)
#         transformed = self.gcn_transform(item_emb) * weights
#         return torch.cat([transformed, torch.zeros_like(transformed)], dim=1)
#
#
# class CDECF(torch.nn.Module):
#     def __init__(self, config):
#         super(CDECF, self).__init__()
#         self.config = config
#         self.num_users = config['num_users']
#         self.num_items = config['num_items']
#         self.latent_dim = config['latent_dim']
#         self.use_cuda = config['use_cuda']
#         self.device = torch.device("cuda" if self.use_cuda else "cpu")
#
#         # Эмбеддинги как в оригинальном MLP
#         self.embedding_user = nn.Embedding(self.num_users, self.latent_dim)
#         self.embedding_item = nn.Embedding(self.num_items, self.latent_dim)
#
#         # Neural ODE компонент (замена FC слоям MLP)
#         self.ode_func = ODEFunc(self.latent_dim, config['layers'])
#         # Increased number of time steps for better accuracy
#         self.time_steps = torch.linspace(0, 1, steps=10)
#         if self.use_cuda:
#             self.time_steps = self.time_steps.to(self.device)
#
#         # Финальные слои с улучшенной архитектурой
#         self.affine_output = nn.Sequential(
#             nn.Linear(self.latent_dim * 2, self.latent_dim),
#             nn.ReLU(),
#             nn.Linear(self.latent_dim, 1)
#         )
#         self.logistic = nn.Sigmoid()
#
#         # Инициализация весов
#         if config['weight_init_gaussian']:
#             for m in self.modules():
#                 if isinstance(m, nn.Embedding):
#                     nn.init.normal_(m.weight, 0, 0.01)
#                 elif isinstance(m, nn.Linear):
#                     nn.init.xavier_normal_(m.weight)
#                     if m.bias is not None:
#                         nn.init.constant_(m.bias, 0)
#
#     def forward(self, user_indices, item_indices):
#         user_emb = self.embedding_user(user_indices)
#         item_emb = self.embedding_item(item_indices)
#
#         # Объединение эмбеддингов для ODE
#         ode_input = torch.cat([user_emb, item_emb], dim=1)
#
#         # Make sure time_steps is on the same device as ode_input
#         if self.time_steps.device != ode_input.device:
#             self.time_steps = self.time_steps.to(ode_input.device)
#
#         # Fix the ODE solver configuration to avoid duplicate parameters
#         ode_output = odeint(
#             self.ode_func,
#             ode_input,
#             self.time_steps,
#             method='dopri5',  # Adaptive Runge-Kutta 4(5)
#             options={
#                 'max_num_steps': 1000  # Maximum number of steps
#             }
#         )[-1]
#
#         # Разделяем обновлённые эмбеддинги
#         user_emb_updated, _ = ode_output.chunk(2, dim=1)
#
#         # Конкатенация и предсказание
#         vector = torch.cat([user_emb_updated, item_emb], dim=1)
#         logits = self.affine_output(vector)
#         return self.logistic(logits)
#
#     # Загрузка весов из GMF
#     def load_pretrain_weights(self):
#         gmf_model = GMF(self.config)
#         if self.config['use_cuda']:
#             gmf_model.cuda()
#         resume_checkpoint(gmf_model, self.config['pretrain_mf'], self.config['device_id'])
#         self.embedding_user.weight.data = gmf_model.embedding_user.weight.data
#         self.embedding_item.weight.data = gmf_model.embedding_item.weight.data
#
#
# class CDECFEngine(Engine):
#     def __init__(self, config):
#         self.model = CDECF(config)
#         if config['use_cuda']:
#             use_cuda(True, config['device_id'])
#             self.model.cuda()
#         super(CDECFEngine, self).__init__(config)
#
#         if config['pretrain']:
#             self.model.load_pretrain_weights()





class CDECF(nn.Module):
    def __init__(self, config, user_item_interactions):
        super(CDECF, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.device = torch.device(f"cuda:{config['device_id']}" if config['use_cuda'] else "cpu")

        # Строим матрицу смежности
        self.adj_matrix = self._build_joint_adjacency(user_item_interactions).to(self.device)
        self.adj_matrix = self._normalize_adj(self.adj_matrix)

        # Эмбеддинги
        self.user_emb = nn.Embedding(self.num_users, self.latent_dim).to(self.device)
        self.item_emb = nn.Embedding(self.num_items, self.latent_dim).to(self.device)

        # Neural ODE
        self.ode_func = ODEFunc(self.latent_dim, self.adj_matrix, self.num_users).to(self.device)
        self.time_steps = torch.linspace(0, config['ode_time'], steps=config['ode_steps'], device=self.device)

        self._init_weights()

    def _build_joint_adjacency(self, user_item_sparse):
        """Строит объединенную разреженную матрицу смежности"""
        num_nodes = self.num_users + self.num_items
        rows, cols = user_item_sparse.indices()
        values = user_item_sparse.values()

        # Смещаем индексы товаров
        cols = cols + self.num_users

        # Транспонированные индексы
        trans_rows = cols
        trans_cols = rows

        # Объединяем индексы
        all_rows = torch.cat([rows, trans_rows])
        all_cols = torch.cat([cols, trans_cols])
        all_values = torch.cat([values, values])

        # Создаем объединенную sparse матрицу
        indices = torch.stack([all_rows, all_cols])
        return torch.sparse_coo_tensor(
            indices,
            all_values,
            size=(num_nodes, num_nodes)
        ).coalesce()

    def _normalize_adj(self, adj):
        """Нормировка разреженной матрицы смежности"""
        rowsum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(rowsum, -0.5).view(-1)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to_sparse()

        # Умножение разреженных матриц
        tmp = torch.sparse.mm(adj, d_mat_inv_sqrt)
        return torch.sparse.mm(d_mat_inv_sqrt, tmp)

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, 0, 0.01)
        nn.init.normal_(self.item_emb.weight, 0, 0.01)

    def forward(self, users, items):
        users = users.to(self.device)
        items = items.to(self.device)
        E_u = self.user_emb(users)
        E_i = self.item_emb(items)

        ode_input = torch.cat([E_u, E_i], dim=1)
        ode_output = odeint(
            self.ode_func,
            ode_input,
            self.time_steps,
            method=self.config['ode_solver']
        )[-1]

        E_u_final, E_i_final = ode_output.chunk(2, dim=1)

        predictions = torch.sum(E_u_final * E_i_final, dim=1)
        predictions = torch.sigmoid(predictions)
        return predictions

        # return E_u_final * E_i_final  # [batch_size, latent_dim]

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        if self.config['weight_decay'] > 0:
            l2_reg = (self.user_emb.weight.norm() + self.item_emb.weight.norm()) * self.config['weight_decay']
            loss += l2_reg
        return loss

    def load_pretrain_weights(self):
        """Загрузка предобученных весов для GMF части"""
        if self.config['pretrain_mf']:
            print(f"Loading pretrained weights from {self.config['pretrain_mf']}")
            pretrained_model = torch.load(self.config['pretrain_mf'])
            self.user_emb.weight.data.copy_(pretrained_model['user_emb.weight'])
            self.item_emb.weight.data.copy_(pretrained_model['item_emb.weight'])


class CDECFEngine(Engine):
    def __init__(self, config):
        # Загружаем взаимодействия пользователь-товар
        user_item_interactions = self._load_interactions(config)

        self.model = CDECF(config, user_item_interactions)
        if config['use_cuda']:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(CDECFEngine, self).__init__(config)

        if config['pretrain']:
            self.model.load_pretrain_weights()

    def _load_interactions(self, config):
        """
        Загружает взаимодействия пользователь-товар в виде sparse матрицы
        """
        # Здесь нужно реализовать загрузку ваших данных
        # Пример для случайных данных:
        indices = torch.randint(0, config['num_items'], (config['num_users'] * 10,))
        rows = torch.randint(0, config['num_users'], (config['num_users'] * 10,))
        values = torch.ones(config['num_users'] * 10)

        return torch.sparse_coo_tensor(
            torch.stack([rows, indices]),
            values,
            size=(config['num_users'], config['num_items'])
        ).coalesce()

class ODEFunc(nn.Module):
    def __init__(self, latent_dim, adj_matrix, num_users):
        super().__init__()
        self.latent_dim = latent_dim
        self.adj_matrix = adj_matrix  # оставляем разреженной!
        self.num_users = num_users
        self.device = adj_matrix.device

        self.weight_generator = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, t, x):
        # x размерности (batch_size, latent_dim * 2)

        batch_size = x.size(0)
        E_u, E_i = x[:, :self.latent_dim], x[:, self.latent_dim:]

        # Теперь работаем только с эмбеддингами батча

        # Формируем эмбеддинги полного графа (пользователи + айтемы)
        full_embeddings = torch.zeros(self.num_users + (self.adj_matrix.size(0) - self.num_users),
                                       self.latent_dim, device=self.device)

        # Заполняем эмбеддинги только юзеров и айтемов из батча
        full_embeddings_batch_indices = torch.cat([
            torch.arange(0, E_u.size(0), device=self.device),
            torch.arange(self.num_users, self.num_users + E_i.size(0), device=self.device)
        ])
        full_embeddings[full_embeddings_batch_indices] = torch.cat([E_u, E_i], dim=0)

        # Разреженное матричное произведение
        graph_effect = torch.sparse.mm(self.adj_matrix, full_embeddings)

        # Только батч-узлы:
        graph_effect_u = graph_effect[:E_u.size(0)]  # пользователи
        graph_effect_i = graph_effect[self.num_users:self.num_users + E_i.size(0)]  # айтемы

        # Вычитаем self-loop влияние
        effect_u = graph_effect_u - E_u
        effect_i = graph_effect_i - E_i

        # Генерация весов
        weights = self.weight_generator(x)

        # Применение весов
        dE_u = weights * effect_u
        dE_i = weights * effect_i

        return torch.cat([dE_u, dE_i], dim=1)

