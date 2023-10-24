# Margaret Li
# 1/26/23
# This program loads the preexisting model and predicts probabilities

# import necessary libraries
import pandas as pd
import numpy as np
import tensorflow_decision_forests as tfdf
import tensorflow as tf
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# read data in
protein_info = pd.read_csv("light_and_dark_protein_original.tsv", sep='\t')

# Make a copy of the dataframe so we have a training version and a original_version
all_protein_info = protein_info.copy()

### Select features for training with ANN ###
features=['molecular_weight', 'gravy', 'pI', 'rna_detected_percent', 'highest_tpm']

#### Renormalize the values so that the most useful range is between 0.0 and 1.0
protein_info.loc[ :, 'molecular_weight'] = protein_info['molecular_weight'] / 100
protein_info.loc[ protein_info[ 'molecular_weight'] > 1, 'molecular_weight'] = 1.0
protein_info.loc[ :, 'gravy'] = ( protein_info['gravy'] + 2 ) / 4
protein_info.loc[ :, 'pI'] = ( protein_info['pI'] - 4 ) / 8
protein_info.loc[ :, 'rna_detected_percent'] = protein_info['rna_detected_percent'] / 100
protein_info.loc[ :, 'highest_tpm'] = protein_info['highest_tpm'] / 300
protein_info.loc[ protein_info[ 'highest_tpm'] > 1, 'highest_tpm'] = 1.0

# Make another copy post-normalized
normalized_all_protein_info = protein_info.copy()

# Select just the canonical for training
protein_info = protein_info[ (protein_info['status'] == 'canonical') | (protein_info['status'] == 'not observed') ]

#### Create a set of 0s and 1s for the labels
protein_info['binary_status'] = protein_info.loc[:,'status'].apply(lambda x: 1 if x == 'canonical' else 0)
normalized_all_protein_info['binary_status'] = normalized_all_protein_info.loc[:,'status'].apply(lambda x: 1 if x == 'canonical' else 0)

## define classification parameters 
features_and_label = features.copy()
features_and_label.append('binary_status')
X = protein_info[features_and_label]
y = protein_info['binary_status']
all_X = normalized_all_protein_info[features_and_label]

# random state is specified so each split will be the same
# not totally necessary since split happens before the for loop
X_train_large, X_test_large, y_train, y_test = train_test_split(X, y,
                                                    train_size = 0.75, 
                                                    test_size = 0.25,
                                                    random_state = 42)


## ROC Curve
# go through each feature, train, make predictions, and plot
# ROC curve for this feature. Afterwards, go through the 
# save process for all features

X_train = X_train_large
X_test = X_test_large


# turn test and train sets to be tensorflow dfs
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_train, label="binary_status")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X_test, label="binary_status")
all_ds = tfdf.keras.pd_dataframe_to_tf_dataset(all_X, label="binary_status")

print("Loading stored model..")
model_sub = tf.keras.models.load_model('light_and_dark_predicted_DF.tf')

print("Classifier summary:")
model_sub.summary()


# Generate model predictions for the test set
predictions = list((model_sub.predict(test_ds, verbose=0)))

## sketch ROC curve for this feature or all features
probabilities_list = list(predictions)

## start the graphic for ROC curve. For loop below will add curves to it
plt.figure()
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.grid(True)

fpr, tpr, thresholds = roc_curve(np.ravel(list(y_test)), np.ravel(probabilities_list))
roc_auc = auc(fpr, tpr)
feature = 'all'
plt.plot(fpr,tpr,lw=2,label=feature+" (AUC %0.2f)" % roc_auc)
plt.legend(loc="lower right")

#plt.show()
plt.savefig('Figure_ML_DF_allfeatures.pdf',format='pdf')
plt.savefig('Figure_ML_DF_allfeatures.svg',format='svg')
plt.close()

# Perform model predictions on all data
predictions = model_sub.predict(all_ds, verbose=0)

## Append the prodictions column and write to output file
all_protein_info['predicted_prob'] = predictions
all_protein_info.to_csv('light_and_dark_predicted_DF.tsv', sep="\t", index=False)


#### Write out the decision tree
with open('Figure_ML_DF_Model_Tree.html', 'w') as outfile:
    print(tfdf.model_plotter.plot_model(model_sub, tree_idx=0, max_depth=5), file=outfile)


#### Plot out the feature importance
inspector = model_sub.make_inspector()

# show all the available variable importances to the model
print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
    print("\t", importance)

# create figure
plt.figure(figsize=(12, 4))

# Mean decrease in AUC of the class 1 vs the others.
variable_importance_metric = "SUM_SCORE"
variable_importances = inspector.variable_importances()[variable_importance_metric]

# Extract the feature name and importance values.
# `variable_importances` is a list of <feature, importance> tuples.
feature_names = [vi[0].name for vi in variable_importances]
feature_importances = [vi[1] for vi in variable_importances]
# The feature are ordered in decreasing importance value.
feature_ranks = range(len(feature_names))

# scale importance values
max_value = feature_importances[0]
for i in range(0, len(feature_importances)):
    feature_importances[i] = round(feature_importances[i] / max_value, 2)

# generate bar graph
bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
plt.yticks(feature_ranks, feature_names)
plt.gca().invert_yaxis()

# label each bar with values
for importance, patch in zip(feature_importances, bar.patches):
    plt.text(patch.get_x() + patch.get_width() + 0.003, patch.get_y() + 0.3, f"{importance:.2f}", va="top")

plt.xlabel('Relative feature importance')
plt.tight_layout()

plt.show()
#plt.savefig('Figure_ML_DF_feature_importance.pdf',format='pdf')
#plt.savefig('Figure_ML_DF_feature_importance.svg',format='svg')
#plt.close()

