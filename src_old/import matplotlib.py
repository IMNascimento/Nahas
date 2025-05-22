import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

# --- Função auxiliar para avaliar um modelo ---
def evaluate_model(model, X: torch.Tensor, y: torch.Tensor, title: str = ""):
    model.eval()
    with torch.no_grad():
        probs = model(X).cpu().numpy().ravel()
    preds = (probs >= 0.5).astype(int)
    y_true = y.cpu().numpy().ravel()
    
    # 1) Matriz de confusão
    cm = confusion_matrix(y_true, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    
    # 2) Curva ROC
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    disp.plot(ax=axes[0], cmap=plt.cm.Blues)
    axes[0].set_title(f"{title} Confusion Matrix")
    
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[1].set_title(f"{title} ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()
    
    # 3) Classification report
    print(f"{title} Classification Report:\n", classification_report(y_true, preds))


# --- Avaliar o modelo original ---
evaluate_model(model, X_train_tensor, y_train_tensor, title="Train (no L2)")
evaluate_model(model, X_test_tensor,  y_test_tensor,  title="Test  (no L2)")

# --- Retrain com L2 regularization ---
from torch import optim

model_l2 = LogisticRegressionModel(input_dim)
optimizer_l2 = optim.SGD(model_l2.parameters(), lr=0.01, weight_decay=0.01)
criterion     = nn.BCELoss()

num_epochs = 1000
for epoch in range(1, num_epochs + 1):
    model_l2.train()
    optimizer_l2.zero_grad()
    outputs = model_l2(X_train_tensor)
    loss    = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer_l2.step()

# --- Avaliar o modelo com L2 ---
evaluate_model(model_l2, X_train_tensor, y_train_tensor, title="Train (with L2)")
evaluate_model(model_l2, X_test_tensor,  y_test_tensor,  title="Test  (with L2)")
