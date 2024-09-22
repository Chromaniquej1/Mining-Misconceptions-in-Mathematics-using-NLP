# Mining-Misconceptions-in-Mathematics-using-NLP

In this project, developed an NLP model driven by ML to accurately predict the affinity between misconceptions and incorrect answers (distractors) in multiple-choice questions. This solution will suggest candidate misconceptions for distractors, making it easier for expert human teachers to tag distractors with misconceptions.

## Overview
This project aims to develop a machine learning model that predicts misconceptions associated with multiple-choice question answers. The model is designed to assist educators by identifying potential misconceptions that students may have based on their selected answers.

## Project Structure
The project is organized into several modules for better maintainability and readability:

- `data_preprocessing.py`: Handles the loading and cleaning of data, including text processing and vectorization.
- `model.py`: Defines the neural network architecture used for prediction.
- `train.py`: Contains functions to train the model on the preprocessed data.
- `evaluate.py`: Implements evaluation metrics to assess model performance.
- `test.py`: Integrates the overall workflow, including data loading, training, and evaluation.
- `main.py`: The entry point of the application that runs the entire pipeline.

## Requirements
- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn
