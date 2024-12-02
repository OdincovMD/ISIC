import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random
from classifier import Classifier

def evaluate_classifier(image_folder_path: str,
                        labels_csv_path: str,
                        classifier,
                        preprocess_fn=None) -> dict:
    """
    Evaluate any image classifier by comparing its predictions with ground truth labels.

    Parameters:
    -----------
    image_folder_path : str
        Path to the folder containing images
    labels_csv_path : str
        Path to CSV file with true labels
    classifier : object
        Any classifier object that implements predict() method
    preprocess_fn : callable, optional
        Function to preprocess images before passing to classifier

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Load the labels
    labels_df = pd.read_csv(labels_csv_path)
    labels_df.set_index('image_name', inplace=True)

    # Lists to store predictions and true labels
    predictions = []
    true_labels = []
    failed_images = []

    # Process each image
    for image_name in labels_df.index:
        image_path = os.path.join(image_folder_path, image_name)
        try:

            pred = classifier.predict(image_path)
            # Store prediction and true label\
            predictions.append(pred)
            true_labels.append(labels_df.loc[image_name, 'benign_malignant'])

        except Exception as e:
            failed_images.append((image_name, str(e)))
            continue

    # Calculate metrics
    metrics = {}
    metrics['classification_report'] = classification_report(true_labels, predictions)
    metrics['confusion_matrix'] = confusion_matrix(true_labels, predictions)
    metrics['accuracy'] = (np.array(predictions) == np.array(true_labels)).mean()
    metrics['failed_images'] = failed_images

    return metrics


def visualize_results(metrics: dict) -> None:
    """
    Visualize evaluation results.

    Parameters:
    -----------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    # Print classification report
    print("Classification Report:")
    print("-" * 60)
    print(metrics['classification_report'])
    print(f"\nOverall Accuracy: {metrics['accuracy']:.2%}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Print failed images if any
    if metrics['failed_images']:
        print("\nFailed to process images:")
        for img, error in metrics['failed_images']:
            print(f"- {img}: {error}")


# Test the implementation
if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    classifier = Classifier(device='cpu', voting='soft')

    # Evaluate
    results = evaluate_classifier(
        image_folder_path='/home/hardbox/python/ISIC/test/images',  # Path doesn't matter for random classifier
        labels_csv_path='/home/hardbox/python/ISIC/test/labels.csv',
        classifier=classifier,
    )

    # Visualize results
    visualize_results(results)