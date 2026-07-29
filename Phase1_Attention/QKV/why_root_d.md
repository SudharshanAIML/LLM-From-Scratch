# Let's do a small experiment.

Suppose the raw attention scores are:

[2, 3, 1]

After Softmax:

≈ [0.24, 0.67, 0.09]

Nice.

The model still considers all three tokens.

Now suppose the scores become larger because our vectors are 768-dimensional:

[200, 210, 190]

What does Softmax do?

Remember:

Softmax = e^x / summation(e^x)

Look at the exponentials:

e^200
e^210
e^190

Even though the scores differ by only 10 or 20, exponentials explode.

The probabilities become approximately:

[0.000045, 0.999955, ~0]

Instead of saying:

"Token 2 is somewhat more important."

Softmax says:

"ONLY TOKEN 2 MATTERS. Ignore everything else."

Why is that bad?

Imagine the sentence:

The cat sat on the mat.

Suppose the word "sat" should attend to:

cat (40%)
on (35%)
mat (25%)

A healthy attention distribution:

cat  : 0.40
on   : 0.35
mat  : 0.25

But if the scores become huge:

cat  : 0.999
on   : 0.001
mat  : 0.000

The model becomes too confident.

It stops combining information from multiple words.

Attention becomes almost like picking a single word.


# Google's Brilliant Fix

Instead of changing the dot product, they simply divide by:

d
	​


So the attention scores become:

d
	​

QK
T
	​


For GPT-2:

d = 64

So:

√64 = 8

If the dot product is:

80

It becomes:

80 / 8 = 10

Much more manageable for Softmax.

But why √d and not d?

This is where the paper gets clever.

Imagine each element of Q and K:

has mean ≈ 0
has variance ≈ 1

Each product:

Qi × Ki

has variance around 1.

Now we sum d of them:

Q·K = Q₁K₁ + Q₂K₂ + ... + Q_dK_d

When you sum d independent random variables, the variance becomes approximately:

Variance ≈ d

The standard deviation is:

√Variance = √d

So the typical size of the dot product grows like √d, not like d.

Dividing by √d keeps the distribution of attention scores roughly the same regardless of the embedding dimension.

That's the statistical reason behind the scaling factor.