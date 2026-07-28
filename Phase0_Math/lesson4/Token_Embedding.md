# token embedding
Embeddings: How Does GPT Understand Token IDs?

# mental model
          Animal District

      Dog
       ●

 Cat ●      Wolf ●

            Fox ●


----------------------------

         Vehicle District

Car ●

Truck ●

Bus ●

# 🧠 Assessment
Question 1

Why can't GPT use token IDs directly?

Explain it using your own words, not mine.
Question 2

Why is one-hot encoding better than token IDs, but still not enough?
Question 3

Suppose I randomly shuffle the embedding vectors like this:

Dog → Car's vector
Car → Apple's vector
Apple → Dog's vector

What do you think would happen to GPT's understanding of language?

Why?
🌟 Research Challenge

Imagine I give you two choices for the initial embedding matrix:

Option A

All vectors are initialized to:

[0, 0, 0, ..., 0]

Option B

All vectors are initialized with small random numbers.

Which one would you choose, and why?