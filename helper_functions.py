from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib


def custom_metric(y_true, y_pred, custom_weight=0.7):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    combined_score = (custom_weight * recall) + ((1 - custom_weight) * precision)
    return combined_score


def calculate_metrics(actual_result, predicted_result):
    results = {
        "Accuracy": accuracy_score(actual_result, predicted_result),
        "Precision": precision_score(actual_result, predicted_result),
        "Recall": recall_score(actual_result, predicted_result),
        "Score": custom_metric(actual_result, predicted_result)
    }
    print(results)
    return results
