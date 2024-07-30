# pip install tiktoken
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = "Hello, do you like coffee? In the sunlit terraces of someunknownPlace."

if __name__ == '__main__':
    # Encode text
    ids = tokenizer.encode(text, allowed_special={""})
    print(ids)
    # [15496, 11, 466, 345, 588, 6891, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]

    # Decode text
    decoded_txt = tokenizer.decode(ids)
    print(decoded_txt)
    # Hello, do you like coffee? In the sunlit terraces of someunknownPlace.
