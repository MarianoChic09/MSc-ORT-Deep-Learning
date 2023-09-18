from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def eval_model(training_history, model, test_X, test_y, field_name):
    """
    Model evaluation: plots, classification report
    @param training: model training history
    @param model: trained model
    @param test_X: features 
    @param test_y: labels
    @param field_name: label name to display on plots
    """
    ## Trained model analysis and evaluation
    f, ax = plt.subplots(2,1, figsize=(5,5))
    ax[0].plot(training_history.history['loss'], label="Loss")
    ax[0].plot(training_history.history['val_loss'], label="Validation loss")
    ax[0].set_title('%s: loss' % field_name)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Accuracy
    ax[1].plot(training_history.history['accuracy'], label="Accuracy")
    ax[1].plot(training_history.history['val_accuracy'], label="Validation accuracy")
    ax[1].set_title('%s: accuracy' % field_name)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # Accuracy by category
    test_pred = model.predict(test_X)
    
    test_pred = np.argmax(test_pred, axis=1)
    test_truth = np.argmax(test_y, axis=1)
    df = pd.DataFrame()
    df['y_test'] = test_pred
    df['Class'] = test_truth
    df['Correct'] = test_pred==test_truth
    acc_by_category = df.groupby('Class').agg({'Correct':'mean'})
    acc_by_category['Correct'].plot(kind='bar', title='Accuracy by %s' % field_name, colormap = 'Paired')
    plt.ylabel('Accuracy')
    plt.show()

    # Print metrics
    print("Classification report")
   
    print(metrics.classification_report(test_truth, test_pred, target_names=list(map(lambda x: str(x),np.unique(test_truth)))))
    print("Confusion matrix")
    print(metrics.confusion_matrix(test_truth, test_pred))
    print("")
    print("")
    # Loss function and accuracy
    test_res = model.evaluate(test_X, test_y, verbose=0)
    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])