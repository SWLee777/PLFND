import os
from utils.dataloader import bert_data
from models.FastPLFND import Trainer as FastPLFNDTrainer

class Run():
    def __init__(self, config):
        self.configinfo = config

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']

        self.train_path = self.root_path + 'train.pkl'
        self.val_path = self.root_path + 'val.pkl'
        self.test_path = self.root_path + 'test.pkl'

        self.category_dict = {
            "科技": 0,
            "军事": 1,
            "教育考试": 2,
            "灾难事故": 3,
            "政治": 4,
            "医药健康": 5,
            "财经商业": 6,
            "文体娱乐": 7,
            "社会生活": 8
        }

    def get_dataloader(self):
        loader = bert_data(max_len=self.max_len, batch_size=self.batchsize, vocab_file=self.vocab_file,
                           category_dict=self.category_dict, num_workers=self.num_workers)
        train_loader = loader.load_data(self.train_path, True)
        val_loader = loader.load_data(self.val_path, False)
        test_loader = loader.load_data(self.test_path, False)
        return train_loader, val_loader, test_loader

    def main(self):
        train_loader, val_loader, test_loader = self.get_dataloader()
        if self.model_name == 'FastPLFND':
            trainer = FastPLFNDTrainer(
                emb_dim=self.emb_dim,
                mlp_dims=self.mlp_dims,
                bert=self.bert,
                use_cuda=self.use_cuda,
                lr=self.lr,
                train_loader=train_loader,
                dropout=self.dropout,
                weight_decay=self.weight_decay,
                val_loader=val_loader,
                test_loader=test_loader,
                category_dict=self.category_dict,
                early_stop=self.early_stop,
                epoches=self.epoch,
                save_param_dir=os.path.join(self.save_param_dir, self.model_name),
            )
        trainer.train()
        return trainer.test(val_loader)
