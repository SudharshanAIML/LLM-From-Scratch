sentence  = "I love AI"
tokens = sentence.split(" ")
print(tokens)

#vocabs matrix

vocabs = {
    "I" : 1,
    "love" : 2,
    "AI" : 3
}

ids = [vocabs[token] for token in tokens]
print(ids)

#simple tokenizer