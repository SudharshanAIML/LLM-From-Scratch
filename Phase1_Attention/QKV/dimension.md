
# why N^2
# What Does This 3×3 Matrix Mean?

This is the most important visualization.

Rows = Who is asking? (Query)

Columns = Who is being looked at? (Key)

Imagine the result is:

        I    love    AI

I      2.1   1.5    0.7

love   1.8   2.9    2.4

AI     0.4   2.6    3.2

Read one row at a time.

First row
I

↓

How much does "I"

care about

"I", "love", "AI"?

Second row:

love

↓

How much does "love"

care about

"I", "love", "AI"?

Third row:

AI

↓

How much does "AI"

care about

"I", "love", "AI"?

🤯 This is why attention is O(N²).

Not because of vocabulary size.

Because every token compares itself with every other token.

If there are N tokens:

N × N comparisons


# QUESTION:
Let's Pause Here

You've just derived:

Q = XWQ

K = XWK

V = XWV

QKᵀ

without memorizing anything.

Now I want to check your understanding before we continue.

🎯 Challenge

Suppose the sentence has 100 tokens.

The embedding size is 768.

Each attention head uses 64-dimensional queries, keys, and values.

Tell me the shapes of:

X
WQ
Q
K
QKᵀ   

<!-- 100x768
768x64
100x64
100x64
100x100 -->