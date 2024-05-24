import sys
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import fill_missing_values, drop_column, transform_altitude

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv)) 

#in an ideal world this would validated
model = sys.argv[1]
X_test_path = sys.argv[2]
y_test_path = sys.argv[3]

# load the model from disk
loaded_model = pickle.load(open(model, 'rb'))
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

#feature eng on test data
print("Feature engineering")
X_test = transform_altitude(X_test)
X_test = drop_column(X_test, col_name='Unnamed: 0')
X_test = drop_column(X_test, col_name='Quakers')
X_test = fill_missing_values(X_test)

y_test_pred = loaded_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
print (f"MSE on test is: {mse_test}")



def adjusted_r_squared(r_squared, X):
    adjusted_r2 = 1 - ((1 - r_squared) * (len(X) - 1) / (len(X) - X.shape[1] - 1))
    return adjusted_r2





'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Plot formatting
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18

def evaluate_model_class(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
    
    baseline = {}
    
    baseline['recall'] = r2_score(test_labels, 
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
    
    for metric in ['mean_squared_error', 'mean_absolute_error', 'r2']:
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

evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)
plt.savefig('image/nb1_roc_auc_curve.png')
'''