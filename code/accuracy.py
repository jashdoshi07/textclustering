import pandas as pd
import numpy as np


class confusion_matrix():
    def __init__(self, predicted):
        self.pred = predicted

    def print(self):
        actuals = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2]

        #print(len(actuals))

        act = pd.Series(actuals, name="Actual")
        pred = pd.Series(self.pred, name="Predicted")

        df_confusion = pd.crosstab(act, pred)

        print(df_confusion)

        #print(type(df_confusion))

        #print(df_confusion.columns)
        columns = [0,1,2]
        sum = df_confusion[columns].sum(axis=0)
        #print(sum)
        original = [8,8,8]

        df_matrix = df_confusion.to_numpy()
        #print(len(df_matrix))

        tp = 0
        fp = 0
        fn = 0
        for i in range(len(df_matrix)):
            for j in range(len(df_matrix)):
                if i == j:
                    tp = tp + df_matrix[i][j]
                if i !=j:
                    fp = fp + df_matrix[i][j]


            if i != 0:
                fn = fn +df_matrix[i][0]




        #print(tp, fp, fn)

        fn = fp

        precision = tp/(tp + fp)
        print("Precision will be: ", precision)

        recall = tp/(tp + fn)
        print("Recall will be", recall)

        f1 = 2*((precision*recall)/(precision + recall))

        print("f1 score will be", f1)

        return df_confusion


