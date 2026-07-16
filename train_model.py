from model_pipeline import train_and_save_model

if __name__ == "__main__":
    summary = train_and_save_model()
    print("Model training completed successfully.")
    print(summary)
