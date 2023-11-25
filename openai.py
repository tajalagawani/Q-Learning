
import os
import numpy as np
import random
import openai

# Define your OpenAI API key (preferably as an environment variable)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Q-Learning settings
n_states = 10  # Number of different types of questions
n_actions = 5  # Number of answers to choose from for each question
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # Exploration rate
n_episodes = 1000  # Number of training rounds

# Initialize Q-table
Q = np.zeros((n_states, n_actions))

# Define function to choose the next action
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, n_actions - 1)
        print(f"[Exploration] Choosing random action {action} for state {state}")
    else:
        action = np.argmax(Q[state, :])
        print(f"[Exploitation] Choosing best action {action} for state {state}")
    return action

# Define function to update the Q-table
def update_Q(state, action, reward):
    predict = Q[state, action]
    target = reward + discount_factor * np.max(Q[state, :])
    Q[state, action] += learning_rate * (target - predict)
    print(f"Updated Q-table at state {state}, action {action} with reward {reward}")

# Function to interact with the GPT model via the chat API
def get_LLM_answers(question):
    print(f"Fetching answers for the question: '{question}'")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Replace with the appropriate model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        )
        print(f"Full response from OpenAI: {response}")
        if 'error' in response:
            print(f"Error from OpenAI: {response['error']['message']}")
            return None
        else:
            answers = [msg['content'] for msg in response['choices'][0]['message'] if msg['role'] == 'assistant']
            print(f"Received answers: {answers}")
            return answers
    except openai.error.OpenAIError as e:
        print(f"OpenAI API error occurred: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Function for evaluating the chosen answer
def evaluate_answer(answer, question):
    # Modify this logic based on how you want to evaluate the answer
    reward = 1 if answer and 'correct' in answer else 0
    print(f"Evaluating answer '{answer}' for question '{question}', Reward: {reward}")
    return reward

# Function to generate or fetch a new question
def generate_question():
    questions = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the primary colors?",
        "Describe the process of photosynthesis.",
        "Who wrote 'To Kill a Mockingbird'?"
    ]
    question = random.choice(questions)
    state = questions.index(question) % n_states
    print(f"Generated question: '{question}', State: {state}")
    return question, state

# Main Q-learning loop
for episode in range(n_episodes):
    print(f"\nStarting episode {episode + 1}")
    question, state = generate_question()
    answers = get_LLM_answers(question)
    
    if not answers:
        print("No answers received, skipping this episode.")
        continue

    action = choose_action(state)
    if action >= len(answers):
        print(f"Action {action} is out of range. Adjusting to fit the available answers.")
        action = len(answers) - 1

    chosen_answer = answers[action]
    reward = evaluate_answer(chosen_answer, question)
    update_Q(state, action, reward)

# Output the final Q-table
print("\nTraining completed. Final Q-table:")
print(Q)
