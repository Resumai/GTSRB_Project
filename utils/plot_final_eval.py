import matplotlib.pyplot as plt

def plot_final_eval(val_acc, final_test_acc):
    labels = ['Validation Accuracy', 'Final Test Accuracy']
    values = [val_acc * 100, final_test_acc * 100]

    plt.figure(figsize=(7, 6))  # Taller figure
    bars = plt.bar(labels, values, color=['skyblue', 'lightcoral'])
    plt.ylim(0, 105)  # Add breathing room at top

    # Add text labels above bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f'{yval:.2f}%', ha='center', fontsize=12)

    plt.title("Validation vs Final Test Accuracy", fontsize=14)
    plt.ylabel("Accuracy (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout(pad=2) 
    plt.show()

# Example usage:
plot_final_eval(val_acc=0.9814, final_test_acc=0.8766)
