# Lesson
Bytes vs characters.
UTF-8.
Byte-level BPE.
Why GPT-2 never needs <UNK>.


# 🎯 Assessment
Question 1

Why did GPT-2 choose bytes instead of characters as the starting point for BPE?

Don't just say "because there are 256 bytes."

Explain the engineering advantage.
Question 2

Suppose tomorrow Unicode adds 10,000 new emojis.

Would GPT-2's tokenizer need to be retrained?

Why or why not?
Question 3

Consider these two approaches:

Approach A

Text
↓
Characters
↓
BPE

Approach B

Text
↓
UTF-8
↓
Bytes
↓
BPE

Which one is more universal, and what trade-off does it introduce before BPE learns useful merges?
🌟 Research Challenge

Imagine you're designing a tokenizer for a future model that must process:

    Natural language

    Source code

    DNA sequences (ACGT)

    Mathematical equations

    Music notation

Would you start from:

    Characters?

    Bytes?

    Something else?

Defend your choice like you're presenting it to an engineering team.
