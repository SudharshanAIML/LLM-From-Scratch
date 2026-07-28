# Lesson1
Why tokenization is needed.
Character vs word vs subword tokenization.
Vocabulary explosion.
Unknown words.
Tokenizer fertility.

## Problem
What problem were we trying to solve?
optimal tokenizer

## Failed Solution
Why didn't the previous method work?
they tonkenize per word kind of things now we tokenize thorugh BPE (byte pair encoding) because it tonkenize by frequency

## New Idea
What was invented?
BPE

## Why It Works
Explain in your own words.

## Modern Alternatives
How do today's models improve this?

# 🧠 Assessment

I don't want definitions. I want reasoning.
Question 1

Why can't GPT simply use Unicode numbers as inputs?
Question 2

Suppose I create a tokenizer where every entire word is one token.

What are three serious problems this creates?

Don't just list them—explain why they matter.
Question 3

Why are subword tokens a better compromise than both character-level and word-level tokenization?
Question 4 (Research Level)

Suppose English has 100,000 words, but your vocabulary size is only 30,000 tokens.

How is GPT still able to generate words that are not present in the vocabulary?

Walk me through the full process—from input text to generated output.
⭐ Bonus Challenge

You're the CTO of OpenAI.

You can choose only one tokenizer:

    Character-level

    Word-level

    Subword-level

Which one would you choose, and how would you justify that decision to your engineering team?
