import numpy as np
from sklearn import metrics
import sklearn.metrics as metrics
import pandas as pd
from sklearn.metrics import roc_auc_score

def calculate_roc_auc(actual_class, pred_class, average = "weighted"):
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]

        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    list_values = [v for v in roc_auc_dict.values()]
    average = np.average(list_values)
    return average

def calculate_metrics(y_true, results, le):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(results, axis=1)


    y_true = le.inverse_transform(y_true)
    y_pred = le.inverse_transform(y_pred)
    acc = P = R = F1 = AUC = 0.0
    report = ""
    AUC = calculate_roc_auc(y_true, y_pred)
    try:
        acc = metrics.accuracy_score(y_true, y_pred)
        P = metrics.precision_score(y_true, y_pred, average="weighted")
        R = metrics.recall_score(y_true, y_pred, average="weighted")
        F1 = metrics.f1_score(y_true, y_pred, average="weighted")
        report = metrics.classification_report(y_true, y_pred)

    except Exception as e:
        print (e)
        pass

    return AUC,acc, P, R, F1, report


def format_conf_mat(y_true,y_pred,le):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)


    y_true = le.inverse_transform(y_true)
    y_pred = le.inverse_transform(y_pred)

    conf_mat = pd.crosstab(np.array(y_true), np.array(y_pred), rownames=['gold'], colnames=['pred'], margins=True)
    pred_columns = conf_mat.columns.tolist()
    gold_rows = conf_mat.index.tolist()
    conf_mat_str = ""
    header = "Pred\nGold"
    for h in pred_columns:
        header = header + "\t" + str(h)
    conf_mat_str = header + "\n"
    index = 0
    for r_index, row in conf_mat.iterrows():
        row_str = str(gold_rows[index])
        index += 1
        for col_item in row:
            row_str = row_str + "\t" + str(col_item)
        conf_mat_str = conf_mat_str + row_str + "\n"
    return conf_mat_str


