from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, class_names=None, plot_confusion=True):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')  # or 'weighted'
    
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    if plot_confusion:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()
    
    return acc, f1





def plot_model_comparison(results_dict):
    models = list(results_dict.keys())
    accuracy = [results_dict[m]["accuracy"] for m in models]
    f1 = [results_dict[m]["f1"] for m in models]

    x = range(len(models))
    plt.bar(x, accuracy, width=0.4, label="Accuracy", align='center')
    plt.bar([i + 0.4 for i in x], f1, width=0.4, label="F1 Score", align='center')
    plt.xticks([i + 0.2 for i in x], models)
    plt.ylabel("Score")
    plt.title("Model Accuracy vs F1 Score")
    plt.legend()
    plt.tight_layout()
    plt.show()
