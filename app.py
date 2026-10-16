import os
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from src.dataset import IrisDataset, FEATURES
from src.model import LogisticRegressionModel

def get_data_path():
    paths = ["data/iris.csv", "iris.csv", "../data/iris.csv"]
    for p in paths:
        if os.path.exists(p):
            return p
    return "data/iris.csv"

def explore_data():
    csv_path = get_data_path()
    df = pd.read_csv(csv_path)
    stats = df.describe().to_string()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="sepal_length", y="sepal_width", hue="target", palette="viridis", ax=ax)
    ax.set_title("Sepal Length vs Sepal Width")
    plt.close(fig)
    
    return stats, fig

def train_model(test_size, max_iter):
    try:
        # Convertir asegurando que maneje comas o puntos
        if isinstance(test_size, str):
            test_size = float(test_size.replace(',', '.'))
        else:
            test_size = float(test_size)
            
        csv_path = get_data_path()
        dataset = IrisDataset(test_size=test_size, random_state=42)
        X_train, X_test, y_train, y_test = dataset.get_data(csv_path)
        
        model = LogisticRegressionModel(max_iter=int(max_iter), random_state=42)
        model.train(X_train, y_train)
        
        train_acc = model.model.score(X_train, y_train)
        test_acc = model.model.score(X_test, y_test)
        
        os.makedirs("models", exist_ok=True)
        model.save("models/model.joblib")
        
        return f"Training completed successfully.\n- Train Accuracy: {train_acc:.4f}\n- Test Accuracy: {test_acc:.4f}"
    except Exception as e:
        return f"Error during training: {e}"

def evaluate_model():
    try:
        csv_path = get_data_path()
        dataset = IrisDataset(test_size=0.25, random_state=42)
        _, X_test, _, y_test = dataset.get_data(csv_path)
        
        model = LogisticRegressionModel()
        model.load("models/model.joblib")
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix (Accuracy: {acc:.2f})")
        plt.close(fig)
        
        return f"Evaluation loaded successfully. Test Accuracy: {acc:.4f}", fig
    except Exception as e:
        return f"Error loading model. Please train the model first in the previous tab. Details: {e}", None

with gr.Blocks() as demo:
    gr.Markdown("# Practice 4: Interactive Demo - Iris Machine Learning")
    gr.Markdown("This application allows data exploration, model training, and performance evaluation.")
    
    with gr.Tabs():
        with gr.TabItem("Data Exploration"):
            gr.Markdown("### Dataset Statistics and Visualization")
            btn_explore = gr.Button("Load Data and Plot")
            out_stats = gr.Textbox(label="Descriptive Statistics", lines=10)
            out_plot = gr.Plot(label="Feature Plot")
            btn_explore.click(fn=explore_data, outputs=[out_stats, out_plot])
            
        with gr.TabItem("Training Interface"):
            gr.Markdown("### Adjust Hyperparameters and Train Model")
            slider_test = gr.Slider(minimum=0.1, maximum=0.5, value=0.25, step=0.05, label="Test Size")
            slider_iter = gr.Slider(minimum=50, maximum=500, value=200, step=50, label="Max Iterations")
            btn_train = gr.Button("Train Model")
            out_train = gr.Textbox(label="Training Results")
            btn_train.click(fn=train_model, inputs=[slider_test, slider_iter], outputs=[out_train])
            
        with gr.TabItem("Model Evaluation"):
            gr.Markdown("### Performance and Confusion Matrix")
            btn_eval = gr.Button("Evaluate Current Model")
            out_eval_text = gr.Textbox(label="Metrics")
            out_eval_plot = gr.Plot(label="Confusion Matrix")
            btn_eval.click(fn=evaluate_model, outputs=[out_eval_text, out_eval_plot])

if __name__ == "__main__":
    demo.launch()