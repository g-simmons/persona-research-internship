#!/usr/bin/env python3


import json
import ast


term_freq = {}
with open('term__frequencies/wordnet-frequencies.json', 'r') as f:
    term_freq = json.load(f)

wordnet_freq = term_freq["wordnet.txt"]

print(wordnet_freq["entity"])