#!/usr/bin/env python3

import zstandard as zstd
import re
import os
import time


dctx = zstd.ZstdDecompressor()

def get_word_count(word: str) -> int:
    input_dir = "/mnt/bigstorage/raymond/pile-uncopyrighted/train/"
    for filename in os.listdir(input_dir):
        print(filename)
        with open(input_dir + filename, 'rb') as ifh:
            counter = 0
            for chunk in dctx.read_to_iter(ifh, read_size=1):
                # print(chunk)
                words = re.split(r' |\\\\n\\\\n|\\\\u|. |, |; |: |! ', str(chunk))
                for i in words:
                    if i == word:
                        print(i)
                        counter += 1
                        print(counter)
    return counter
                

start_time = time.time()
a = get_word_count("the")
b = get_word_count("more")
c = get_word_count("dog")
d = get_word_count("animal")


print(a)
print(b)
print(c)   
print(d)
print("--- %s seconds ---" % (time.time() - start_time))