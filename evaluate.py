from sklearn.metrics import accuracy_score, f1_score, hamming_loss, precision_score, recall_score

def evaluate_model(y_true, y_pred):
    print("Overall Evaluation Metrics:")
    
    total_accuracy, total_f1, total_precision, total_recall, total_hamming = 0, 0, 0, 0, 0
    num_metrics = y_true.shape[1]

    # Calculate metrics for each output
    for i in range(num_metrics):
        accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
        f1 = f1_score(y_true[:, i], y_pred[:, i], average='weighted', zero_division=0)
        precision = precision_score(y_true[:, i], y_pred[:, i], average='weighted', zero_division=0)
        recall = recall_score(y_true[:, i], y_pred[:, i], average='weighted', zero_division=0)
        hamming = hamming_loss(y_true[:, i], y_pred[:, i])
        
        total_accuracy += accuracy
        total_f1 += f1
        total_precision += precision
        total_recall += recall
        total_hamming += hamming

    # Calculate average scores
    avg_accuracy = total_accuracy / num_metrics
    avg_f1 = total_f1 / num_metrics
    avg_precision = total_precision / num_metrics
    avg_recall = total_recall / num_metrics
    avg_hamming = total_hamming / num_metrics

    print("\nAverage Scores Across All Misconceptions:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average Hamming Loss: {avg_hamming:.4f}")
