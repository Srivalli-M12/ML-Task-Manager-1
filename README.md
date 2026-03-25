# ML-Task-Manager-1
A command-line Task Management Application built with Python. This project serves as a standard to-do list with a twist: it integrates a **Machine Learning text classification model** to automatically predict whether a new task should be marked as High, Medium, or Low priority based on its description.

## 🚀 Features
* **Auto-Predict Priority:** Uses a trained Naive Bayes classifier to read a task description and automatically assign the correct priority.
* **Manual Entry:** Allows users to manually input tasks and priorities to help train the model.
* **Task Recommendation:** Suggests a random high-priority task from the current to-do list to help users focus.
* **Persistent Storage:** Saves all tasks to a local `tasks.csv` file so data is never lost between sessions.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Data Handling:** Pandas
* **Machine Learning:** Scikit-Learn (`CountVectorizer`, `MultinomialNB`)

## 🧠 How the Machine Learning Works
This app uses Natural Language Processing (NLP) to classify text. 
1. **Vectorization:** `CountVectorizer` converts the text descriptions into a matrix of token counts (a vocabulary).
2. **Classification:** A `MultinomialNB` (Naive Bayes) model analyzes the probability of certain words belonging to High, Medium, or Low priorities based on the existing `tasks.csv` dataset.
3. **Dynamic Learning:** Every time a new task is added or removed, the model automatically retrains itself, getting smarter as the dataset grows.

## 💻 How to Run the App

**1. Clone the repository:**
```bash
git clone [https://github.com/Srivalli-M12/ML-Task-Manager-1.git](https://github.com/Srivalli-M12/ML-Task-Manager-1.git)
cd ML-Task-Manager-1
```
**2. Install dependencies:**
```bash
pip install -r requirements.txt
```
**3. Run the application:**
```bash
python main.py
```
