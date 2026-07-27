# Simulating a tiny language model

probabilities = {
    "blue": 0.91,
    "beautiful": 0.05,
    "dark": 0.03,
    "pizza": 0.01
}

print("Input : The sky is")

prediction = max(probabilities, key=probabilities.get)

print("Predicted next token:", prediction)