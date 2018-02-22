import re
from re import *
import regexv.utils

vec_master = regexv.utils.vecMaster()


def search(pattern, string, flags=0, delimiters=('<', '>')):
    for substring in re.finditer(r"(?:{}([^{}]+){})*".format(delimiters[0], delimiters[0], delimiters[1]), pattern):
        if not substring.group(1):
            continue

        m = substring.group(1)
        words = m.split(",")
        words = [word.strip() for word in words]

        expanded_words = vec_master.neighbor_expansion(words)

        first_word_index = pattern.find(words[0])
        last_word_index = pattern.find(words[-1]) + len(words[-1])

        expanded_string = pattern[:first_word_index]
        expanded_string += "|".join(expanded_words)
        expanded_string += pattern[last_word_index:]
        pattern = expanded_string

    pattern = re.sub(delimiters[0], '(', pattern)
    pattern = re.sub(delimiters[1], ')', pattern)
    return re.search(pattern, string, flags=flags)

# def wrap(method):
#     def fn(*args, **kwargs):
#         return method(*args, **kwargs)
#     return fn
