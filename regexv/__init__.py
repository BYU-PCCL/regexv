import re
from re import *
import types

import regexv
import regexv.utils

vec_master = regexv.utils.vecMaster()

def expand(pattern, delimiters=('<', '>')):
    for substring in re.finditer(r"(?:{}([^{}]+){})*".format(delimiters[0], delimiters[0], delimiters[1]), pattern):
        if not substring.group(1):
            continue

        m = substring.group(1)
        words = m.split(",")
        words = [word.strip() for word in words]

        expanded_words = vec_master.neighbor_expansion(words)

        first_word_index = pattern.find(m)
        last_word_index = first_word_index + len(m)

        expanded_string = pattern[:first_word_index]
        expanded_string += "|".join(list(expanded_words))
        expanded_string += pattern[last_word_index:]
        pattern = expanded_string

    pattern = re.sub(delimiters[0], '(', pattern)
    pattern = re.sub(delimiters[1], ')', pattern)
    return pattern


def compile_file(from_file, to_file):
    # read lines from from_file
    content = from_file.readlines()
    for line in content:

        # if line is regex, then expand
        if line[0] != "#" and line[0] != "[":
            line = expand(line)

        # write to new file
        to_file.write(line)


def wrap(method):
    def fn(*args, **kwargs):
        kwargs.setdefault('delimiters', ('<', '>'))
        delimiters = kwargs['delimiters']
        pattern = expand(args[0], delimiters=delimiters)

        new_args = list(args)
        new_args[0] = pattern
        del kwargs['delimiters']
        return method(*new_args, **kwargs)

    return fn


for item in dir(re):
    method = re.__dict__.get(item)
    if isinstance(method, types.FunctionType):
        regexv.__dict__[item] = wrap(method)

