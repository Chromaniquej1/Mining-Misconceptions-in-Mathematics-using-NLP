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


## Usage

To run the entire project, use the following command:

```python main.py```
This will execute the following steps:

Load and preprocess the data: Data is cleaned and transformed into a suitable format for model training.
Train the machine learning model: The model is trained using the preprocessed data.
Evaluate the model's performance: Various metrics are calculated to assess how well the model predicts misconceptions.
Data Format
The expected input data should be in CSV format with the following columns:

QuestionId_Answer: The identifier for each question and answer option.
MisconceptionId: A list of associated misconception IDs for each answer. The values should be space-separated integers representing different misconceptions.
