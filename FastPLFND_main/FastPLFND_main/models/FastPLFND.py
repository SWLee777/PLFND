from .layers import *
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import tqdm
import time
from transformers import BertModel
from utils.utils import data2gpu, Averager, Recorder, metrics
from torch.optim.lr_scheduler import StepLR

class FastPLFND(nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, dropout):
        super(FastPLFND, self).__init__()
        self.domain_num = 9
        self.num_expert = 5

        self.bert = BertModel.from_pretrained(bert, from_tf=True).requires_grad_(False)

        self.feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.experts = nn.ModuleList([ConvFeatureExtractor(self.feature_kernel, emb_dim) for _ in range(self.num_expert)])

        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, mlp_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dims[-1], self.num_expert),
            nn.Softmax(dim=1)
        )

        self.attention = AttentionLayer(emb_dim)
        self.domain_embedder = nn.Embedding(num_embeddings=self.domain_num, embedding_dim=emb_dim)
        self.classifier = FeedForwardNN(320, mlp_dims, dropout)
        self.prob_classifier = nn.Sequential(
            nn.Linear(emb_dim, self.num_expert),
            nn.Dropout(dropout),
            nn.Softmax(dim=1)
        )

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        category = kwargs['category']

        init_features = self.bert(inputs, attention_mask=masks)[0]
        features, _ = self.attention(init_features, masks)

        idxs = torch.tensor([index for index in category]).view(-1, 1).cuda()
        domain_embeddings = self.domain_embedder(idxs).squeeze(1)

        gate_input = torch.cat([domain_embeddings, features], dim=-1)
        gate_values = self.gate(gate_input)

        expert_outputs = torch.stack([exp(init_features) for     exp in self.experts], dim=1)
        combined_weights = gate_values.unsqueeze(2) * self.prob_classifier(features).unsqueeze(2)
        expert_contributions = expert_outputs * combined_weights

        shared_features = torch.sum(expert_contributions, dim=1)
        label_pred = self.classifier(shared_features)

        return torch.sigmoid(label_pred.squeeze(1)), shared_features

class SensitivityModel(nn.Module):
    def __init__(self, input_dim, mlp_dims, dropout):
        super(SensitivityModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, mlp_dims[-1]),
            nn.BatchNorm1d(mlp_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dims[-1], 1)
        )

    def forward(self, shared_features):
        return self.classifier(shared_features)


class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 bert,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 loss_weight=[1, 0.006, 0.009, 5e-5],
                 early_stop=10,
                 epoches=100):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert = bert
        self.dropout = dropout

        if not os.path.exists(save_param_dir):
            os.makedirs(save_param_dir)
        self.save_param_dir = save_param_dir

    def train(self):
        self.model = FastPLFND(self.emb_dim, self.mlp_dims, self.bert, self.dropout)
        self.sensitivity_model = SensitivityModel(input_dim=320, mlp_dims=self.mlp_dims, dropout=self.dropout)
        if self.use_cuda:
            self.model = self.model.cuda()
            self.sensitivity_model = self.sensitivity_model.cuda()
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.sensitivity_model.parameters()),
                                      lr=self.lr, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.98)

        for epoch in range(self.epoches):
            self.model.train()
            self.sensitivity_model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                if self.use_cuda:
                    batch_data = {k: v.cuda() for k, v in batch_data.items()}
                label = batch_data['label'].unsqueeze(1)
                optimizer.zero_grad()

                label_pred, shared_features = self.model(**batch_data)
                sensitivity_pred = self.sensitivity_model(shared_features)
                loss = loss_fn(sensitivity_pred, label.float())

                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                avg_loss.add(loss.item())

            print('Training Epoch {}; Loss {}; '.format(epoch + 1, avg_loss.item()))

            results = self.test(self.val_loader)
            mark = recorder.add(results)

            model_save_path = os.path.join(self.save_param_dir, f'model_epoch_{epoch + 1}.pkl')
            torch.save(self.model.state_dict(), model_save_path)
            sensitivity_model_save_path = os.path.join(self.save_param_dir, f'sensitivity_model_epoch_{epoch + 1}.pkl')
            torch.save(self.sensitivity_model.state_dict(), sensitivity_model_save_path)
            print(f'Model saved at: {model_save_path}')
            print(f'Sensitivity Model saved at: {sensitivity_model_save_path}')

            if mark == 'save':
                best_model_path = os.path.join(self.save_param_dir, f'parameter_FastPLFND_best.pkl')
                torch.save(self.model.state_dict(), best_model_path)
                best_sensitivity_model_path = os.path.join(self.save_param_dir,
                                                           f'sensitivity_parameter_FastPLFND_best.pkl')
                torch.save(self.sensitivity_model.state_dict(), best_sensitivity_model_path)
            elif mark == 'esc':
                break

        # 进行裁剪
        self.model = prune_model(self.model)
        self.sensitivity_model = prune_model(self.sensitivity_model)

        self.model.load_state_dict(torch.load(best_model_path))
        self.sensitivity_model.load_state_dict(torch.load(best_sensitivity_model_path))
        results = self.test(self.test_loader)
        print(results)
        return results, best_model_path

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        self.model.eval()
        self.sensitivity_model.eval()
        data_iter = tqdm.tqdm(dataloader)
        start_time = time.time()
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                if self.use_cuda:
                    batch_data = {k: v.cuda() for k, v in batch_data.items()}
                batch_label = batch_data['label']
                batch_category = batch_data['category']

                label_pred, shared_features = self.model(**batch_data)
                sensitivity_pred = self.sensitivity_model(shared_features)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(torch.sigmoid(sensitivity_pred).detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        end_time = time.time()  # Record the end time for inference
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.2f} seconds")

        return metrics(label, pred, category, self.category_dict)


def prune_model(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.8)
            prune.remove(module, 'weight')
    return model