import os
from utils.dataloader import BertData
from models.plfend import Trainer as PLFENDTrainer

class Run:
    def __init__(self, config):
        self.configinfo = config

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.learning_rate = config['lr']
        self.batch_size = config['batchsize']
        self.embedding_dim = config['emb_dim']
        self.max_length = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert_model_path = config['bert']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout_rate = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.num_epochs = config['epoch']
        self.save_param_dir = config['save_param_dir']

        self.train_path = os.path.join(self.root_path, 'train.pkl')
        self.val_path = os.path.join(self.root_path, 'val.pkl')
        self.test_path = os.path.join(self.root_path, 'test.pkl')

        self.category_dict = {
            "科技": 0,
            "军事": 1,
            "教育考试": 2,
            "灾难事故": 3,
            "政治": 4,
            "医药健康": 5,
            "财经商业": 6,
            "文体娱乐": 7,
            "社会生活": 8,
        }

    def get_dataloader(self):
        loader = BertData(
            max_len=self.max_length,
            batch_size=self.batch_size,
            vocab_file=self.vocab_file,
            category_dict=self.category_dict,
            num_workers=self.num_workers
        )

        train_loader = loader.load_data(self.train_path, shuffle=True)
        val_loader = loader.load_data(self.val_path, shuffle=False)
        test_loader = loader.load_data(self.test_path, shuffle=False)
        return train_loader, val_loader, test_loader

    def main(self):
        train_loader, val_loader, test_loader = self.get_dataloader()
        if self.model_name == 'plfend':
            # Pass all necessary parameters, including the new ones
            trainer = PLFENDTrainer(
                embedding_dim=self.embedding_dim,
                mlp_dims=self.mlp_dims,
                bert_model_path=self.bert_model_path,
                use_cuda=self.use_cuda,
                learning_rate=self.learning_rate,
                train_loader=train_loader,
                dropout_rate=self.dropout_rate,
                weight_decay=self.weight_decay,
                val_loader=val_loader,
                test_loader=test_loader,
                category_dict=self.category_dict,
                early_stop_threshold=self.early_stop,
                num_epochs=self.num_epochs,
                param_save_dir=os.path.join(self.save_param_dir, self.model_name),
            )
            trainer.train()

