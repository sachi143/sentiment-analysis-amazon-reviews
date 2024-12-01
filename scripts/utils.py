import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate the model and save the confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Ensure the outputs directory exists
    output_dir = '../outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the confusion matrix plot
    plt.savefig(f'{output_dir}/confusion_matrix_{model_name}.png')
    plt.close()

    # Print classification report
    print(f"\n{model_name} Classification Report:\n")
    print(classification_report(y_true, y_pred))
