# 🧠 Assessment
Question 1

Why is initializing every embedding to zeros a bad idea?

Explain it using your own intuition.

Question 2

Suppose these two sentences appear millions of times during training:

Cats drink milk.
Dogs drink milk.

Why do you expect the embeddings for Cat and Dog to become more similar over time?

Question 3

Suppose GPT has never seen the sentence:

Dogs enjoy pizza.

Could it still make a reasonable prediction after the word "Dogs"?

Why?

Don't answer with "because it's trained." Think in terms of embeddings and context.

🌟 Research Challenge

Here's one to really think about.

Suppose I secretly swap the names of every animal in the training data:

Dog ↔ Car

Cat ↔ Bus

Wolf ↔ Bicycle

Everything else in the corpus stays internally consistent.

Would GPT still learn a coherent language model?

Or would training fail?