import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score

np.seterr(all='raise')

class iNode():
    def __init__(self, value=None, left=None, right=None, split_att=None, split_value=None):
        self.value= value
        self.left = left
        self.right = right
        self.split_att = split_att
        self.split_value = split_value

#Isolation Forest
class iForest():
    def __init__(self, x_train, x_test, y_train, y_test, num_trees=100, subsample_size=256):
        self.train_data = x_train
        self.test_data = x_test
        self.train_label = y_train
        self.test_label = y_test
        self.num_trees = num_trees

        if subsample_size > x_train.shape[0]:
            self.subsample_size = 0
        else:
            self.subsample_size = subsample_size
        
        self.pred_label = None
        self.models = None

    #generate the isolation subtrees through recursion
    def i_tree(self, data, hight, h_limit):
        #generate the external node when the iForest height reaches the limit or only 1 instance left
        if hight >= h_limit or data.shape[0] <= 1:
            return iNode(value=data.shape[0])
        
        #generate the internal node
        else:
            #randomly select 1 attribute and 1 value of it to split the dataset
            random_select = data.sample(n=1, axis=1)
            split_att = random_select.columns[0]
            split_value = random_select.sample(n=1).iloc[0,0]

            l_data = data[data[split_att]<split_value]
            r_data = data[data[split_att]>=split_value]

            l_tree = self.i_tree(l_data, hight+1, h_limit)
            r_tree = self.i_tree(r_data, hight+1, h_limit)

            return iNode(left=l_tree, right=r_tree, split_att=split_att, split_value=split_value)
    
    #generate mutiple isolation trees
    def fit_transform(self):
        forests = []
        if self.subsample_size:
            h_limit = np.floor(np.log2(self.subsample_size))
        else:
            h_limit = np.floor(np.log2(self.train_data.shape[0]))

        for i in range(self.num_trees):
            if not self.subsample_size:
                sub_data = self.train_data
            else:
                sub_data = self.train_data.sample(n=self.subsample_size, replace=False)
            forests.append(self.i_tree(sub_data, 0, h_limit))
        
        self.models = forests
        joblib.dump(forests, "iForests.joblib.dat")
        return forests

    #formula of the average path length of unsuccessful search in BST
    def calc_avg(self, n):
        try:
            return 2*(np.log(n-1)+0.5772156649)-(2*(n-1)/n)
        except:
            return 1

    #calculate the average path length of an instance
    def path_length(self, ins, node, length):
        if node.value is not None:
            return length + self.calc_avg(node.value)
        
        else:
            split_att = node.split_att
            if ins.loc[split_att] < node.split_value:
                return self.path_length(ins, node.left, length+1)
            else:
                return self.path_length(ins, node.right, length+1)

    #evaluate the models
    def predict(self, model_path=None, sort=False):
        #calculate the average path length of reaching external nodes
        test_size = self.test_data.shape[0]
        avg_path_len = self.calc_avg(test_size)

        #load models
        if model_path is not None:
            self.models = joblib.load(model_path)

        #calculate the average path length for each instance
        anom_score = np.zeros([self.test_data.shape[0], ])
        for each_tree in self.models:
            prep_test = lambda x : self.path_length(x, each_tree, 0)
            ins_avg_length = self.test_data.apply(prep_test, axis=1)
            anom_score += np.power(2, -ins_avg_length/avg_path_len)

        anom_score /= len(self.models)
        self.pred_label = anom_score

        #sort in the descending order so that the top n instances are the top n anomalies
        if sort:
            anom_score = pd.Series(anom_score).sort_values(ascending=False)

        return pd.Series(anom_score)
    
    #calculate AUC
    def auc(self):
        if self.pred_label is None:
            self.predict()
        return roc_auc_score(self.test_label, self.pred_label)
