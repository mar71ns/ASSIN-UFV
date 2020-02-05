from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import datetime
import pydot
from sklearn.tree import export_graphviz
import pandas as pd
import json
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import pprint
from pydash import keys, get, set_, find_key, find_last_key, deburr, values, omit
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import itertools

def load_json(name):
    #filename = "./temporary_files/" + name
    filename = "./" + name
    try:
        if(os.path.isfile(filename)):
            with open(filename, 'r', encoding="utf8") as f:
                return(json.load(f))
                f.close()
        else:
            with open(filename, 'w+', encoding="utf8") as f:
                print("Nao achou")
                json.dump({}, f, indent = 4, sort_keys=True, ensure_ascii=False)
                f.close()
                return {}
    except Exception as e:
        print('Erro ao abrir: '+ filename +"\nError:\n"+ str(e))
        return {}

def save_json(name, data):
    filename = name+'.json'
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=4, sort_keys=False, ensure_ascii=False)
        f.close()

def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, 
                                     [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, 
                                      [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

def overlapping_percentage(x, y):
    return (100.0 * len(set(x) & set(y))) / len(set(x) | set(y))


file_train = "assin2-train"
file_test = "assin2-test"

RSEED = 50
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data_inputs = pd.read_json( file_train + '-processed.json')
data_inputs = data_inputs.transpose()
data_inputs = data_inputs.drop('f1', axis=1)
data_inputs = data_inputs.drop('f2', axis=1)
data_labels = np.array(load_json( file_train + '-labels-classifier.json'))

# # Imputation of missing values
# train = train.fillna(train.mean())
# test = test.fillna(test.mean())

# # Features for feature importances
# features = list(train.columns)

## Descriptive statistics for each column
#print(data_inputs.describe())

# print('Training Features Shape:', train.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape:', test.shape)
#print('Testing Labels Shape:', test_labels.shape)

# Create the model with 100 trees
model1 = RandomForestClassifier(
                        bootstrap=True,
                        criterion='gini',
                        n_estimators=1000,
                        max_features='auto',
                        n_jobs=-1,
                        min_samples_split=16,
                        min_samples_leaf=4,
                        oob_score=True,
                        verbose=1,
                        warm_start=True,
                        random_state=RSEED)

model2 = ExtraTreesClassifier(
                        bootstrap=True,
                        criterion = 'gini',
                        n_estimators = 1000,
                        max_features = 'auto',
                        n_jobs=-1,
                        # min_samples_split = 8,
                        # min_samples_leaf = 2,
                        #oob_score = True,
                        #warm_start = True,
                        random_state = RSEED)

model3 = GradientBoostingClassifier(
                        loss='deviance',
                        criterion='friedman_mse',
                        max_features='sqrt',
                        #learning_rate=0.001,
                        #max_depth=10,
                        #min_samples_split = 8,
                        #min_samples_leaf = 2,
                        #oob_score = True,
                        #verbose = 1,
                        #warm_start = True,
                        random_state=RSEED)

model4 = HistGradientBoostingClassifier(
                        loss='auto',
                        # loss='auto',
                        l2_regularization=1,
                        learning_rate=0.01,
                        # max_iter= 9999,
                        max_bins = 128,
                        max_leaf_nodes  = 8,
                        max_depth  = 16,
                        # min_samples_leaf = 20,
                        # n_iter_no_change = 150,
                        scoring="loss",
                        # verbose = 0,
                        validation_fraction= 0.1,
                        random_state=RSEED)
    
model5 = DecisionTreeClassifier(
        criterion='gini',
        # criterion='entropy',
        splitter='best',  # random|best
        #max_depth=10,
        #min_samples_split = 8,
        #min_samples_leaf = 2,
        random_state=RSEED)

model6 = MLPClassifier(
                        hidden_layer_sizes=(32,64),
                        activation='relu',
                        # activation='logistic',
                        solver='lbfgs',
                        max_iter=2000,
                        learning_rate='adaptive',
                        n_iter_no_change=10,
                        verbose=False,
                    )

# kernel = DotProduct() + WhiteKernel()
kernel = 1.0 * RBF(1.0)
model7 = GaussianProcessClassifier(
    kernel=kernel,
    random_state=RSEED,
    max_iter_predict = 8,
    n_jobs=-1
)

name_base_estimator =  "model1"
base_estimator =        model1
model8 = BaggingClassifier(
        base_estimator=base_estimator,
        n_estimators=16,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=True,
        oob_score=True,
        n_jobs=-1,
        random_state=RSEED,
        verbose=True,
    )

model9 = LogisticRegression(
    random_state=RSEED,
    solver="newton-cg",
    #solver="liblinear",
)
# model9 = LogisticRegressionCV(
#     random_state=RSEED,
#     solver="newton-cg",
#     max_iter= 300,
#     cv=3,
#     n_jobs=-1
# )
# modelos = [ model3 ]
modelos = [ model7, model8 ]
modelos = [ model1, model2, model3, model4, model6, model8 ]
modelos = [ model1, model2, model3, model4, model6, model8, model9 ]

best_result_acc=0
best_result_f1=0
best_result_model=""

for model in modelos:
    #model = model8
    # Fit on training data
    model.fit(data_inputs, data_labels)
    
    #-----------------------
    if (False):
        importances = model.feature_importances_
        # print(importances.shape)
        # print(data_inputs.columns.shape)
        # exit()
        std = np.std([tree.feature_importances_ for tree in model.estimators_],
                    axis=0)
        # indices = np.argsort(importances)[::-1]
        indices = np.argsort(importances)[::-1]
        # Print the feature ranking
        print("Feature ranking:")
        for f in range(data_inputs.shape[1]):
            print("%d. %s (%f)" % (f + 1, data_inputs.columns[indices[f]], importances[indices[f]]))        
    #-----------------------
    test = pd.read_json( file_test + '-processed.json ')
    test = test.transpose()
    test_labels = np.array(load_json( file_test + '-labels-classifier.json' ))
    test = test.drop('f1', axis=1)
    test = test.drop('f2', axis=1)

    predictions = model.predict(test)

    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    
    f1 = f1_score(test_labels, predictions, average='macro')
    print(classification_report(test_labels, predictions))
    classificacao_resultados = classification_report(test_labels, predictions)
    
    matriz_confusao = (confusion_matrix(test_labels, predictions))
    print(matriz_confusao)
    # Print out the mean absolute error (mae)

    # absolute_diff = test_labels - predictions
    # mse = (absolute_diff ** 2).mean()
    # # Calculate mean absolute percentage error (MAPE)
    # # mape = 100 * (errors / test_labels)
    # mape = 1
    # # Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    accuracy = accuracy_score(test_labels, predictions)
    
    model_name = ""
    try:
        model_name = type(model).__name__
    except Exception as e:
        model_name = "? BaggingRegressor - Model 2"
        pass

    print('-'*100)
    print()
    print(model_name)
    print('Similarity evaluation')
    print('F1\t\taccuracy')
    print('-------\t\t------------------')
    print('{:7.3f}\t\t{:18.3f}'.format(f1, (accuracy*100)))
    # print('\nAccuracy:', round(accuracy, 2), '%.')
    # print('Mean Absolute Error:', round(np.mean(errors), 2), '')
    pass
    now = datetime.datetime.now()
    with open(file_test+"-results-classifiers.txt", 'a+', encoding='utf8') as f:
        if f1 > best_result_f1:
            best_result_f1 = f1
            best_result_acc = accuracy
            best_result_model = model_name
        f.write('\n')
        #f.write('Similarity evaluation')
        f.write('-' * 110)
        f.write('\n')
        f.write('\n{:40s} {:s}'.format('Model', model_name))
        f.write('\n{:40s} {:s}'.format('File Train', file_train))
        f.write('\n{:40s} {:s}'.format('File Test', file_test))
        f.write('\n{:40s} {:3.3f}'.format('f1', f1))
        f.write('\n{:40s} {:3.3f}'.format('Accuracy', (accuracy * 100)))
        f.write('\n')

        if file_train is not "assin2-train":    
            f.write('\nMatriz de Confusao')
            f.write('\n{:10s} {:3s} {:3s} {:3s}'.format("", "Non", "Ent", "Par"))
            f.write('\n{:10s} {:3d} {:3d} {:3d}'.format("None", matriz_confusao[0][0], matriz_confusao[0][1], matriz_confusao[0][2]))
            f.write('\n{:10s} {:3d} {:3d} {:3d}'.format("Entailment", matriz_confusao[1][0], matriz_confusao[1][1], matriz_confusao[1][2]))
            f.write('\n{:10s} {:3d} {:3d} {:3d}'.format("Parafrase", matriz_confusao[2][0], matriz_confusao[2][1], matriz_confusao[2][2]))
            f.write('\n')
        else:
            f.write('\nMatriz de Confusao')
            f.write('\n{:10s} {:3s} {:3s}'.format("", "Non", "Ent", "Par"))
            f.write('\n{:10s} {:3d} {:3d} '.format("None", matriz_confusao[0][0], matriz_confusao[0][1]))
            f.write('\n{:10s} {:3d} {:3d} '.format("Entailment", matriz_confusao[1][0], matriz_confusao[1][1]))
            f.write('\n')

        f.write('\nClassification Report\n')
        f.write(classificacao_resultados)
        f.write('\nParams:')
        try:
            f.write(json.dumps(model.get_params(deep=False), indent=4, sort_keys=True,))
            #f.write(json.loads(model.get_params(deep=True)))
        except Exception as e:
            f.write('\n')
            f.write(name_base_estimator)
            f.write(json.dumps(base_estimator.get_params(deep=False), indent=4, sort_keys=True,))
            print(e)
            #f.write(model.get_params(deep=True))
            pass
        f.write('\n')
        f.write(now.strftime("%Y-%m-%d %H:%M:%S"))
        f.write('\n')
        f.write('-' * 110)
        f.close()
print('\nBest Result:')
print('{:s}\t\t{:18.3f}'.format(best_result_model, best_result_f1))

exit()

feature_list = list(data_inputs.columns)
# Import tools needed for visualization
# Pull out one tree from the forest
tree = model.estimators_[5]
# Import tools needed for visualization
# Pull out one tree from the forest
tree = model.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file='tree.dot',
                feature_names=feature_list, rounded=True, precision=1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')



#-------------------------------------------------------------------------------------

exit()
# Write graph to a png file
graph.write_png('tree.png')


n_nodes = []
max_depths = []

# Stats about the trees in random forest
for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
    
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

# Training predictions (to demonstrate overfitting)
train_rf_predictions = model.predict(train)
train_rf_probs = model.predict_proba(train)[:, 1]

# Testing predictions (to determine performance)
rf_predictions = model.predict(test)
rf_probs = model.predict_proba(test)[:, 1]

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18


evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)
plt.savefig('roc_auc_curve.png')


# Confusion matrix
cm = confusion_matrix(test_labels, rf_predictions)
plot_confusion_matrix(cm, classes = ['Poor Health', 'Good Health'],
                      title = 'Health Confusion Matrix')

plt.savefig('cm.png')
