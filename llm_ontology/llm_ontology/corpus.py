#!/usr/bin/env python3

import zstandard as zstd
import re
import os
import time


dctx = zstd.ZstdDecompressor()

def get_word_count(word: str) -> int:
    input_dir = "/mnt/bigstorage/raymond/pile-uncopyrighted/train/"
    for filename in os.listdir(input_dir):
        # print(filename)
        with open(input_dir + filename, 'rb') as ifh:
            total_count = 0
            counter = 0
            print(dctx.read_to_iter(ifh, read_size=32))
            # for chunk in dctx.read_to_iter(ifh, read_size=32):
            #     # print(chunk)
            #     words = re.split(r' |\\\\n\\\\n|\\\\u|. |, |; |: |! ', str(chunk))
            #     total_count += len(words)
            #     print(total_count)
            #     # for i in words:
            #     #     if i == word:
            #     #         print(i)
            #     #         counter += 1
            #     #         print(counter)
    return total_count
                

start_time = time.time()
a = get_word_count("the")



print(a)

print("--- %s seconds ---" % (time.time() - start_time))