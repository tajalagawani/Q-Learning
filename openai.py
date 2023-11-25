import os
import numpy as np
import random
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

n_states = 10
n_actions = 5
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
n_episodes = 101

Q = np.zeros((n_states, n_actions))

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

def get_LLM_answers(question):
    print(f"Fetching answers for the question: '{question}'")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
        )
        
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

def is_relevant(answer, question):
    question_keywords = set(question.lower().split())
    answer_keywords = set(answer.lower().split())
    common_keywords = question_keywords.intersection(answer_keywords)
    return len(common_keywords) > 0

def is_complete(answer):
    return len(answer.split()) > 5

def is_correct(answer, question):
    return is_relevant(answer, question) and is_complete(answer)

def evaluate_answer(answer, question):
    if not answer:
        return 0
    score = 0
    if is_relevant(answer, question):
        score += 1
    if is_complete(answer):
        score += 1
    if is_correct(answer, question):
        score += 2
    return score

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

for episode in range(n_episodes):
    question, state = generate_question()
    answers = get_LLM_answers(question)
    
    if not answers:
        continue

    action = choose_action(state)
    if action >= len(answers):
        action = len(answers) - 1

    chosen_answer = answers[action]
    reward = evaluate_answer(chosen_answer, question)
    update_Q(state, action, reward)

print("Training completed. Final Q-table:")
print(Q)
