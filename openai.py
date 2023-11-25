
import os
import numpy as np
import random
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

n_states = 10  # Number  of questions
n_actions = 5  # Number of answers to choose from for each question
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1  # Exploration rate
n_episodes = 101# Number of training rounds

# Initialize Q-table
Q = np.zeros((n_states, n_actions))

# Define the function to choose the next action
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, n_actions - 1)
        print(f"Choosing random action: {action} for state: {state}")
    else:
        action = np.argmax(Q[state, :])
        print(f"Choosing best action: {action} for state: {state}")
    return action

def update_Q(state, action, reward):
    predict = Q[state, action]
    target = reward + discount_factor * np.max(Q[state, :])
    Q[state, action] += learning_rate * (target - predict)
    print(f"Updated Q-table at state: {state}, action: {action}, reward: {reward}")

# Function to interact with GPT-4 (or the appropriate model) via the chat API
def get_LLM_answers(question):
    print(f"Fetching answers for the question: '{question}'")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Ensure this is a valid model identifier
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        )
        
        # Extracting the response correctly
        if 'choices' in response and len(response['choices']) > 0:
            message_content = response['choices'][0]['message']['content']
            print(f"Received answer: {message_content}")
            return [message_content]
        else:
            print("No valid response in 'choices'")
            return None
    except openai.error.OpenAIError as e:
        print(f"errrrrrrrorrr: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# Function for evaluating the chosen answer
def evaluate_answer(answer, question):
    # Implement the logic to evaluate the answer
    # Example: Check if the answer is non-empty or contains certain keywords
    reward = 1 if answer and 'correct' in answer else 0
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
    return question, state

# Main Q-learning loop
for episode in range(n_episodes):
    question, state = generate_question()
    answers = get_LLM_answers(question)
    
    if not answers:
        continue  # Skip to the next episode if no answers were received

    action = choose_action(state)
    if action >= len(answers):
        action = len(answers) - 1  # Adjust if action is out of range

    chosen_answer = answers[action]
    reward = evaluate_answer(chosen_answer, question)
    update_Q(state, action, reward)

# Output the final Q-table
print("Training completed. Final Q-table:")
print(Q)
