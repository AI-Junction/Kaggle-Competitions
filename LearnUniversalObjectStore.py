# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 20:16:23 2018

@author: echtpar
"""

# Universal Object Storage

import nltk

#nltk.download('punkt')

text = "Water is a transparent, tasteless, odorless, and nearly colorless chemical substance, which is the main constituent of Earth's streams, lakes, and oceans, and the fluids of most living organisms. It is vital for all known forms of life, even though it provides no calories or organic nutrients. Water is found on Earth in large quantities. Water has many properties"

# Sentence tokenizer

doc = nltk.sent_tokenize(text)
for s in doc:
    print(">",s)

print(doc)

# Word tokenizers
    
from nltk import word_tokenize

# Default Tokenisation
tree_tokens = word_tokenize(text)   # nltk.download('punkt') for this

# Other Tokenisers
punct_tokenizer = nltk.tokenize.WordPunctTokenizer()
punct_tokens = punct_tokenizer.tokenize(text)

space_tokenizer = nltk.tokenize.SpaceTokenizer()
space_tokens = space_tokenizer.tokenize(text)

print("DEFAULT: ", tree_tokens)
print("PUNCT  : ", punct_tokens)
print("SPACE  : ", space_tokens)



# Parts of speech (PoS)

nltk.download('averaged_perceptron_tagger')
pos = nltk.pos_tag(tree_tokens)
print(pos)
pos_space = nltk.pos_tag(space_tokens)
print(pos_space)
print(type(pos_space))
print(type(pos_space[0]))

"""

PoS Tag Descriptions 

CC | Coordinating conjunction
 CD | Cardinal number
 DT | Determiner
 EX | Existential there
 FW | Foreign word
 IN | Preposition or subordinating conjunction
 JJ | Adjective
 JJR | Adjective, comparative
 JJS | Adjective, superlative
 LS | List item marker
 MD | Modal
 NN | Noun, singular or mass
 NNS | Noun, plural
 NNP | Proper noun, singular
 NNPS | Proper noun, plural
 PDT | Predeterminer
 POS | Possessive ending
 PRP | Personal pronoun
 PRP\$ | Possessive pronoun RB | Adverb RBR | Adverb, comparative RBS | Adverb, superlative RP | Particle SYM | Symbol TO | to UH | Interjection VB | Verb, base form VBD | Verb, past tense VBG | Verb, gerund or present participle VBN | Verb, past participle VBP | Verb, non-3rd person singular present VBZ | Verb, 3rd person singular present WDT | Wh-determiner WP | Wh-pronoun WP$ | Possessive wh-pronoun
 WRB | Wh-adverb


"""
import numpy as np
#pos_dict = {}
#pos_struct = [[],[2]]
pox = 0
pos_struct = np.array([['a','b']])
print(pos_struct.shape)
for line in doc:
    tree_tokens = word_tokenize(line)   # nltk.download('punkt') for this
    pos = nltk.pos_tag(tree_tokens)
    for l in pos:
        pos_struct = np.append(pos_struct,[[l[0], l[1]]], axis=0)
        #pox = pox + 1

def CreateTree(pos_struct_x):
    #x = 1
    for i in pos_struct_x:
        if i[1] == 
        print(i)

    return


CreateTree(pos_struct)    


print(pos_struct)
print(pos_struct[:,[1]])

print(pos)
for l in pos:
    print([l[0], l[1]])


import re
regex = re.compile("^N.*")
nouns = []
for l in pos:
    if regex.match(l[1]):
        nouns.append(l[0])
print("Nouns:", nouns)


import re
regex = re.compile("^J.*")
adjectives = []
for l in pos:
    if regex.match(l[1]):
        adjectives.append(l[0])
print("Adjectives:", adjectives)


import re
regex = re.compile("^V.*")
verbs = []
for l in pos:
    if regex.match(l[1]):
        verbs.append(l[0])
print("Verbs:", verbs)


import re
regex = re.compile("^R.*")
adverbs = []
for l in pos:
    if regex.match(l[1]):
        adverbs.append(l[0])
print("Adverbs:", adverbs)


import re
regex = re.compile("^C.*")
connectors = []
for l in pos:
    if regex.match(l[1]):
        connectors.append(l[0])
print("Connectors:", connectors)


import re
regex = re.compile("^D.*")
determinants = []
for l in pos:
    if regex.match(l[1]):
        determinants.append(l[0])
print("Determinants:", determinants)







# adjectives, adverbs are decorators of associated nouns
# conjunctions are connectors




porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
snowball = nltk.stem.snowball.SnowballStemmer("english")

print([porter.stem(t) for t in tree_tokens])
print([lancaster.stem(t) for t in tree_tokens])
print([snowball.stem(t) for t in tree_tokens])

sentence2 = "When I was going into the woods I saw a bear lying asleep on the forest floor"
tokens2 = word_tokenize(sentence2)

print("\n",sentence2)
for stemmer in [porter, lancaster, snowball]:
    print([stemmer.stem(t) for t in tokens2])


nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")

wnl = nltk.WordNetLemmatizer()
tokens2_pos = nltk.pos_tag(tokens2)  #nltk.download("averaged_perceptron_tagger")

print([wnl.lemmatize(t) for t in tree_tokens])

print([wnl.lemmatize(t) for t in tokens2])

