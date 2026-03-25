import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random
import os

# Initialize an empty task list
tasks = pd.DataFrame(columns=['description', 'priority'])

# Load pre-existing tasks from a CSV file (if any)
if os.path.exists('tasks.csv'):
    try:
        tasks = pd.read_csv('tasks.csv')
    except pd.errors.EmptyDataError:
        pass # Handle case where the file exists but is completely empty

# Function to save tasks to a CSV file
def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

# Train the task priority classifier
def train_model():
    # Only train if we have data
    if not tasks.empty and len(tasks) >= 2: 
        vectorizer = CountVectorizer()
        clf = MultinomialNB()
        ml_model = make_pipeline(vectorizer, clf)
        ml_model.fit(tasks['description'], tasks['priority'])
        return ml_model
    return None

# Initialize the model on startup
model = train_model()

# Function to add a task to the list
def add_task(description, priority):
    global tasks  # Declare tasks as a global variable
    global model  # Access the global model to retrain it
    
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    if tasks.empty:
        tasks = new_task
    else:
        tasks = pd.concat([tasks, new_task], ignore_index=True)
    
    save_tasks()
    model = train_model() # Retrain the model so it learns from the new task

# Function to remove a task by description
def remove_task(description):
    global tasks # Added missing global declaration
    global model
    
    tasks = tasks[tasks['description'] != description]
    save_tasks()
    model = train_model() # Retrain the model without the removed task

# Function to list all tasks
def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print("\n--- Current Tasks ---")
        print(tasks.to_string(index=False))
        print("---------------------\n")

# Function to recommend a task based on machine learning
def recommend_task():
    if not tasks.empty:
        # Get high-priority tasks
        high_priority_tasks = tasks[tasks['priority'] == 'High']
        
        if not high_priority_tasks.empty:
            # Choose a random high-priority task (converted to list to avoid pandas formatting quirks)
            random_task = random.choice(high_priority_tasks['description'].tolist())
            print(f"\nRecommended task: {random_task} - Priority: High")
        else:
            print("\nNo high-priority tasks available for recommendation.")
    else:
        print("\nNo tasks available for recommendations.")

# Main menu
while True:
    print("\nTask Management App")
    print("1. Add Task (Auto-Predict Priority via ML)")
    print("2. Add Task (Manual Priority)")
    print("3. Remove Task")
    print("4. List Tasks")
    print("5. Recommend Task")
    print("6. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        if model is not None:
            # The model expects a list of strings
            predicted_priority = model.predict([description])[0]
            print(f"Machine Learning assigned Priority: {predicted_priority}")
            add_task(description, predicted_priority)
            print("Task added successfully.")
        else:
            print("Not enough data to predict! Please use Option 2 to add a few tasks manually first.")

    elif choice == "2":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        add_task(description, priority)
        print("Task added successfully.")

    elif choice == "3":
        description = input("Enter exact task description to remove: ")
        remove_task(description)
        print("Task removed successfully.")

    elif choice == "4":
        list_tasks()

    elif choice == "5":
        recommend_task()

    elif choice == "6":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Please select a valid option (1-6).")