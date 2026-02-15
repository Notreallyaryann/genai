import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hello, I am Aryan Patel"
tokens = enc.encode(text)

print("Tokens:", tokens)


Tokens=[13225, 11, 357, 939, 107851, 270, 122760]
decoded = enc.decode(tokens)

print("Decoded Text:", decoded)
