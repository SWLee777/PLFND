import os
import tqdm
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from .layers import CNNExtractor, MaskAttention, MLP
from utils.utils import GpuData, Recorder, Averager, GetMetrics


class PreLocalizationFENDModel(nn.Module):
    def __init__(self, embedding_dim, mlp_dims, bert_model_path, dropout_rate):
        super(PreLocalizationFENDModel, self).__init__()
        self.num_domains = 9
        self.num_experts = 9

        # Sentiment tokenizer and model
        self.sentiment_tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
        self.sentiment_model = BertForSequenceClassification.from_pretrained(
            'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment'
        ).requires_grad_(False)

        # Original BERT model
        self.bert_model = BertModel.from_pretrained(bert_model_path, from_tf=True).requires_grad_(False)

        # Domain-specific pre-trained models
        self.domain_specific_berts = {
            'science': BertModel.from_pretrained('allenai/scibert_scivocab_uncased').requires_grad_(False),
            'biomedical': BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1').requires_grad_(False),
            'general': BertModel.from_pretrained(bert_model_path, from_tf=True).requires_grad_(False)
        }

        # Feature extraction CNN layers
        self.feature_kernels = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.experts = nn.ModuleList([CNNExtractor(self.feature_kernels, embedding_dim) for _ in range(self.num_experts)])

        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(embedding_dim * 3 + self.sentiment_model.config.num_labels, mlp_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dims[-1], self.num_experts),
            nn.Softmax(dim=1)
        )

        # Attention layer
        self.attention_layer = MaskAttention(embedding_dim)
        self.domain_embedding = nn.Embedding(num_embeddings=self.num_domains, embedding_dim=embedding_dim)
        self.classifier = MLP(320, mlp_dims, dropout_rate)
        self.probability_classifier = nn.Sequential(
            nn.Linear(embedding_dim + self.sentiment_model.config.num_labels, self.num_experts),
            nn.Dropout(dropout_rate),
            nn.Softmax(dim=1)
        )

    def forward(self, **kwargs):
        content_inputs = kwargs['content']
        content_masks = kwargs['content_masks']
        domain_category = kwargs['category']

        # Decode inputs using sentiment tokenizer
        decoded_inputs = [self.sentiment_tokenizer.decode(token, skip_special_tokens=True) for token in content_inputs]

        # Initial features from original BERT model
        initial_features = self.bert_model(content_inputs, attention_mask=content_masks)[0]

        # Extract domain-specific features
        domain_features = self._extract_domain_features(content_inputs, content_masks, domain_category)

        # Sentiment features
        sentiment_inputs = self.sentiment_tokenizer(decoded_inputs, return_tensors="pt", padding=True,
                                                    truncation=True).input_ids.cuda()
        sentiment_outputs = self.sentiment_model(sentiment_inputs)
        sentiment_features = sentiment_outputs.logits

        # Attention mechanism
        features, _ = self.attention_layer(initial_features, content_masks)
        features = torch.cat([features, sentiment_features], dim=1)

        # Domain embeddings
        category_indices = torch.tensor([index for index in domain_category], dtype=torch.long).view(-1, 1).cuda()
        domain_embeddings = self.domain_embedding(category_indices).squeeze(1)

        # Gating network input
        gate_input = torch.cat([domain_embeddings, features, domain_features], dim=-1)
        gate_values = self.gate_network(gate_input)

        # Expert outputs and their combination
        expert_outputs = torch.stack([exp(initial_features) for exp in self.experts], dim=1)
        combined_weights = gate_values.unsqueeze(2) * self.probability_classifier(features).unsqueeze(2)
        expert_contributions = expert_outputs * combined_weights

        shared_features = torch.sum(expert_contributions, dim=1)
        label_predictions = self.classifier(shared_features)

        return torch.sigmoid(label_predictions.squeeze(1))

    def _extract_domain_features(self, inputs, masks, categories):
        """
        Extract domain-specific features using pre-trained BERT models.
        """
        device = next(self.parameters()).device
        domain_features = []

        for i, category in enumerate(categories):
            if category == 0:
                bert_model = self.domain_specific_berts['science']
            elif category == 5:
                bert_model = self.domain_specific_berts['biomedical']
            else:
                bert_model = self.domain_specific_berts['general']

            bert_model = bert_model.to(device)
            domain_feature = bert_model(inputs[i:i + 1], attention_mask=masks[i:i + 1])[0]
            domain_features.append(domain_feature)

        domain_features = torch.cat(domain_features, dim=0)
        return domain_features.mean(dim=1)


class Trainer:
    """
    Trainer class for training and evaluating the PrecisionLocalizationFENDModel.
    """
    def __init__(self, embedding_dim, mlp_dims, bert_model_path, use_cuda, learning_rate, dropout_rate, train_loader,
                 val_loader, test_loader, category_dict, weight_decay, param_save_dir, loss_weights=[1, 0.006, 0.009, 5e-5],
                 early_stop_threshold=10, num_epochs=100):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.early_stop_threshold = early_stop_threshold
        self.num_epochs = num_epochs
        self.category_dict = category_dict
        self.loss_weights = loss_weights
        self.use_cuda = use_cuda

        self.embedding_dim = embedding_dim
        self.mlp_dims = mlp_dims
        self.bert_model_path = bert_model_path
        self.dropout_rate = dropout_rate

        if not os.path.exists(param_save_dir):
            os.makedirs(param_save_dir)
        self.param_save_dir = param_save_dir

    def train(self):
        """
        Train the model and save the best-performing model based on validation results.
        """
        self.model = PreLocalizationFENDModel(self.embedding_dim, self.mlp_dims, self.bert_model_path, self.dropout_rate)
        if self.use_cuda:
            self.model = self.model.cuda()
        loss_function = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        recorder = Recorder(self.early_stop_threshold)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()

            for step_n, batch in enumerate(train_data_iter):
                batch_data = GpuData(batch, self.use_cuda)
                labels = batch_data['label']
                if labels.size(0) == 1:  # Skip batches with size 1
                    continue
                optimizer.zero_grad()
                label_predictions = self.model(**batch_data)
                loss = loss_function(label_predictions, labels.float())
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                avg_loss.add(loss.item())

            print(f'Training Epoch {epoch + 1}; Loss {avg_loss.item()};')

            results = self.evaluate(self.val_loader)
            mark = recorder.add(results)

            model_save_path = os.path.join(self.param_save_dir, f'model_epoch_{epoch + 1}.pkl')
            torch.save(self.model.state_dict(), model_save_path)
            print(f'Model saved at: {model_save_path}')

            if mark == 'save':
                best_model_path = os.path.join(self.param_save_dir, 'best_model_params.pkl')
                torch.save(self.model.state_dict(), best_model_path)
            elif mark == 'esc':
                break

        self.model.load_state_dict(torch.load(best_model_path))
        results = self.evaluate(self.test_loader)
        print(results)
        return results, best_model_path

    def evaluate(self, data_loader):
        """
        Evaluate the model on the given data loader.
        """
        predictions = []
        labels = []
        categories = []
        self.model.eval()
        data_iter = tqdm.tqdm(data_loader)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = GpuData(batch, self.use_cuda)
                batch_labels = batch_data['label']
                batch_categories = batch_data['category']
                if batch_labels.size(0) == 1:  # Skip batches with size 1
                    continue
                batch_label_predictions = self.model(**batch_data)

                labels.extend(batch_labels.detach().cpu().numpy().tolist())
                predictions.extend(batch_label_predictions.detach().cpu().numpy().tolist())
                categories.extend(batch_categories.detach().cpu().numpy().tolist())

        return GetMetrics(labels, predictions, categories, self.category_dict)
