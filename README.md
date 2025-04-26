# MovieGenreClassifier

## Task Objectives

This project aims to develop a machine learning model that can predict the genre of a movie based on its textual description. We utilize text classification techniques to categorize movie descriptions into predefined genre labels.

## Data

The project uses the following data files:

* `train_data.txt`: Contains the movie descriptions and their corresponding genre labels used for training the model.
* `test_data.txt`: Contains movie descriptions used for evaluating the model's performance.

## Models

The following machine learning models were implemented:

* Logistic Regression
* Naive Bayes

## Steps to Run the Project

1.  **Clone the repository:**

    ```bash
    git clone <your_repository_url>
    cd MovieGenreClassifier
    ```

2.  **(Optional) Set up a virtual environment (Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required libraries:**

    ```bash
    pip install numpy pandas scikit-learn
    # Any other libraries you used (e.g., fasttext)
    ```

4.  **Run the Jupyter Notebook:**

    * You can run the notebook cell by cell in a Jupyter environment.
    * Alternatively, you can convert the notebook to a Python script and run it.

    ```bash
    jupyter notebook MovieGenreClassifier.ipynb
    ```

    or

    ```bash
    jupyter nbconvert --to script MovieGenreClassifier.ipynb
    python MovieGenreClassifier.py
    ```

5.  **(If you have separate Python scripts):**

    ```bash
    python <your_script_name>.py
    ```

## Code Structure

The code is organized as follows:

* `MovieGenreClassifier.ipynb`: Jupyter Notebook containing the data loading, preprocessing, model training, and evaluation steps.
* (Optional) `data_loading.py`: (If you have separate scripts for functions)
* (Optional) `model_training.py`: (If you have separate scripts for functions)

## Key Functions

* `load_data(file_path)`:  Loads data from a text file.
* `preprocess_text(text)`:  Cleans and preprocesses text data.
* `train_model(X_train, y_train, model_type)`: Trains a specified machine learning model.
* `evaluate_model(model, X_test, y_test)`: Evaluates the trained model.

## Evaluation Criteria

* **Functionality:** The project successfully loads data, trains machine learning models, and predicts movie genres based on descriptions. The Logistic Regression and Naive Bayes models are implemented, and the code provides evaluation metrics (e.g., classification report).
* **Code Quality:** The code is well-structured, easy to read, and includes comments explaining the different steps. The notebook is organized logically, with clear sections for data loading, preprocessing, model training, and evaluation.
* **Documentation:** The README file provides clear instructions on how to run the project and explains the project's objectives and code structure.

## Further Improvements

* Explore other machine learning models (e.g., SVM, FastText).
* Implement hyperparameter tuning to optimize model performance.
* Incorporate more advanced text preprocessing techniques.
* Add more comprehensive error handling and logging.
