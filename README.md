# NLP Emotion Classifier using DistilBERT

This repository contains the implementation of an NLP model that classifies emotions from text using the DistilBERT architecture. The project uses a dataset from Kaggle and includes pre-processing, model training, evaluation, and a Streamlit application for real-time predictions.

## Project Repository

Visit the [GitHub repository](https://github.com/shahabaalam/NLP_Emotion_Classifier.git) to access all files and detailed instructions on setting up and running the project.

## Project Structure

- `NLP_Emotion_Classifier_DistilBERT_KaggleData.ipynb`: Jupyter notebook with detailed steps from data preprocessing to model evaluation.
- `NLP_Emotion_Classifier_DistilBERT.pdf`: A PDF version of the Jupyter notebook.
- `app.py`: Streamlit application for deploying the model as a web service.
- `requirements.txt`: Contains all the necessary Python packages for the project.
- `emotion-detection-model/`: Directory containing the trained DistilBERT model and tokenizer files needed to run the Streamlit application.
  - `config.json`: Configuration file for the model.
  - `model.safetensors`: The trained model file.
  - `special_tokens_map.json`: Special tokens for the tokenizer.
  - `tokenizer_config.json`: Configuration for the tokenizer.
  - `tokenizer.json`: Tokenizer file.
  - `training_args.bin`: Training arguments used during model training.
  - `vocab.txt`: Vocabulary file for the tokenizer.

## Installation

Ensure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

## Running the Streamlit App

To start the web application, run:

```bash
streamlit run app.py
```

Access the web application at the default URL provided by Streamlit (usually `http://localhost:8501`) to interact with the deployed model.

## Usage

Refer to the Jupyter notebook for a step-by-step tutorial on data processing, training the model with Hugging Face's Transformers library, and evaluating the results.

## Model Training

The model is trained using PyTorch with CUDA support, which significantly speeds up the training process. Training details and configurations are stored within `training_args.bin`.

## Contributing

Contributions to this project are welcome. Please fork this repository and submit a pull request to propose your changes.

## License

This project is licensed under the terms of the MIT license.
