from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np


class Recorder():

    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)

def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    for c, res in res_by_category.items():
        try:
            auc_score = roc_auc_score(res['y_true'], res['y_pred'])
            metrics_by_category[c] = {
                'auc': round(auc_score, 4)
            }
        except ValueError as e:
            print(f"Error calculating AUC for category {c}: {str(e)}")
            metrics_by_category[c] = {'auc': 0}

    metrics_by_category['auc'] = round(roc_auc_score(y_true, y_pred, average='macro'), 4)
    y_pred = np.around(np.array(y_pred)).astype(int)
    metrics_by_category['metric'] = round(f1_score(y_true, y_pred, average='macro'), 4)
    metrics_by_category['recall'] = round(recall_score(y_true, y_pred, average='macro'), 4)
    metrics_by_category['precision'] = round(precision_score(y_true, y_pred, average='macro'), 4)
    metrics_by_category['acc'] = round(accuracy_score(y_true, y_pred), 4)

    fscore_list = []

    for c, res in res_by_category.items():
        try:
            y_pred_cat_bin = np.around(np.array(res['y_pred'])).astype(int)
            fscore = round(f1_score(res['y_true'], y_pred_cat_bin, average='macro'), 4)
            metrics_by_category[c] = {
                'precision': round(precision_score(res['y_true'], y_pred_cat_bin, average='macro'), 4),
                'recall': round(recall_score(res['y_true'], y_pred_cat_bin, average='macro'), 4),
                'fscore': round(f1_score(res['y_true'], y_pred_cat_bin, average='macro'), 4),
                'auc': metrics_by_category[c]['auc'],
                'acc': round(accuracy_score(res['y_true'], y_pred_cat_bin), 4)
            }
            fscore_list.append(fscore)
        except Exception as e:
            print(f"Error calculating {c}: {str(e)}")
            metrics_by_category[c] = {
                'precision': 0,
                'recall': 0,
                'fscore': 0,
                'auc': 0,
                'acc': 0
            }

    return metrics_by_category

def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch[0].cuda(),
            'content_masks': batch[1].cuda(),
            'label': batch[2].cuda(),
            'category': batch[3].cuda()
        }
    else:
        batch_data = {
            'content': batch[0],
            'content_masks': batch[1],
            'label': batch[2],
            'category': batch[3]
        }
    return batch_data


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
