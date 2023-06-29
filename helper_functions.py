from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score


def custom_metric(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Define the weight for the custom metric
    custom_weight = 0.7

    combined_score = (custom_weight * recall) + ((1 - custom_weight) * precision)
    return combined_score


def show_results(actual_result, predicted_result, labels):
    cm = confusion_matrix(actual_result, predicted_result, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)
    disp.plot()
    plt.show()
    print(f"Accuracy\t: {accuracy_score(actual_result, predicted_result):.4}")
    print(f"Precision\t: {precision_score(actual_result, predicted_result):.4}")
    print(f"Recall\t\t: {recall_score(actual_result, predicted_result):.4}")
    print(f"Score\t\t: {custom_metric(actual_result, predicted_result):.4}")
