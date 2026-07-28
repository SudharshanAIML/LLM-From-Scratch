# Lesson2
Invented BPE from scratch.
Frequency counting.
Greedy merging.
Compression.
Reusable subwords.

## Problem
What problem were we trying to solve?
invent the BPE (byte pair encoding)

## Failed Solution
Why didn't the previous method work?
because they didn't has the semantic embedding

## New Idea
What was invented?
BPE it was a greedy concept it combines the character which has most frequency as a token

## Why It Works
Explain in your own words.

## Modern Alternatives
How do today's models improve this?


# 🧠 Assessment

I want you to reason, not memorize.
Question 1

Why does BPE start with the smallest possible units instead of whole words?
Question 2

Suppose these words appear in a corpus:

play
player
playing
played
playful

Which character pair do you think is likely to be merged early, and why?

You don't need exact counts—reason it out.
Question 3

BPE is called a greedy algorithm.

What do you think that means in this context?

Can you imagine a situation where choosing the most frequent pair right now might prevent a better overall vocabulary later?
🚀 Research Challenge

Imagine you want to train a tokenizer for medical literature.

Words like:

cardiomyopathy
cardiology
cardiovascular

Should the tokenizer eventually learn:

cardio

as a token?

Why or why not?

Think about compression, reuse, and semantic consistency—not just frequency.


