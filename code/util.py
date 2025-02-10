import os
import psutil
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)  # Memory in MB

def print_metrics(accuracy, f1, cm, train_time, inference_time, memory_usage, path, filename):
    with open(filename, "w") as file:
        file.write("Model Evaluation Metrics\n")
        file.write("========================\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"F1 Score: {f1:.4f}\n")
        file.write(f"Training Time: {train_time:.2f} seconds\n")
        file.write(f"Inference Time: {inference_time:.2f} seconds\n")
        file.write(f"Memory Usage Increase: {memory_usage:.2f} MB\n")
        file.write("\nConfusion Matrix:\n")
        file.write(f"{cm}\n")
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(path, 'confusion_matrix.svg'))

        

