import pandas as pd
import numpy as np
import json

# Menghitung entropy
def calculate_entropy(dataset_label):
    classes,class_counts = np.unique(dataset_label,return_counts = True)
    entropy_value = np.sum([(-class_counts[i]/np.sum(class_counts))*np.log2(class_counts[i]/np.sum(class_counts))
                        for i in range(len(classes))])
    return entropy_value
    
# Menghitung information gain
def calculate_information_gain(dataset,feature,label):
    dataset_entropy = calculate_entropy(dataset[label])  
    values,feat_counts= np.unique(dataset[feature],return_counts=True)
 
    weighted_feature_entropy = np.sum([(feat_counts[i]/np.sum(feat_counts))*calculate_entropy(dataset.where(dataset[feature]
                              ==values[i]).dropna()[label]) for i in range(len(values))])    
    feature_info_gain = dataset_entropy - weighted_feature_entropy
    return feature_info_gain
   
# Buat decision tree
def create_decision_tree (dataset, data, features, label, parent):
  datum = np.unique(data[label], return_counts=True)
  unique_data = np.unique (dataset[label])
 
  if len(unique_data) <= 1:
    return unique_data[0]
  elif len(dataset) == 0:
    return unique_data[np.argmax(datum[1])]
  elif len(features) == 0:
    return parent
  else:
    parent = unique_data[np.argmax(datum[1])]
    item_values = [calculate_information_gain(dataset, feature,label) for feature in features]
    optimum_feature_index = np.argmax(item_values)
    optimum_feature = features[optimum_feature_index]
    decision_tree = {optimum_feature:{}}
    features = [i for i in features if i != optimum_feature]
 
    for value in np.unique (dataset[optimum_feature]):
      min_data = dataset.where(dataset[optimum_feature] == value).dropna()
      min_tree = create_decision_tree (min_data, data, features, label, parent)
      decision_tree [optimum_feature][value] = min_tree
   
    return(decision_tree)
   
def predict(test_data, decision_tree):
  for nodes in decision_tree. keys():
    value = test_data[nodes]
    decision_tree = decision_tree[nodes][value]
    prediction = 0
   
    if type(decision_tree) is dict:
      prediction = predict(test_data, decision_tree)
    else:
      prediction = decision_tree
      break
 
  return prediction
 
def count_error(dataset, decision_tree):
  prediction = []
  count = 0
  len = dataset.shape[0]
 
  for i in range(len):
    pred = predict(dataset.iloc[i, :-1],decision_tree)
    prediction.append(pred)
    if pred != dataset.iloc[i, -1:].values:
      count += 1
 
  dataset['PREDICTION'] = prediction
  err = count / len
  return err
 
dataset = pd.read_csv('jantung.csv')
print(dataset)
 
# Menentukan features dan label
features = dataset.columns[:-1]
label = 'JANTUNG'
parent=None
 
# Buat decision tree dari data features dan label
decision_tree = create_decision_tree(dataset,dataset,features,label,parent)
 
print(json.dumps(decision_tree, indent=4))
 
# Menghitung error
data = dataset.copy()
error = count_error(data, decision_tree)
print('Error: ', error)
data