# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 05:41:00 2018

@author: echtpar
"""

import nltk

nltk.download('punkt')

ulysses = "Mrkgnao! the cat said loudly. She blinked up out of her avid shameclosing eyes, mewing plaintively and long, showing him her milkwhite teeth. He watched the dark eyeslits narrowing with greed till her eyes were green stones. Then he went to the dresser, took the jug Hanlon's milkman had just filled for him, poured warmbubbled milk on a saucer and set it slowly on the floor. — Gurrhr! she cried, running to lap."


# Sentence tokenizer

doc = nltk.sent_tokenize(ulysses)
for s in doc:
    print(">",s)





# Word tokenizers
    
from nltk import word_tokenize

sentence = "Mary had a little lamb it's fleece was white as snow."
# Default Tokenisation
tree_tokens = word_tokenize(sentence)   # nltk.download('punkt') for this

# Other Tokenisers
punct_tokenizer = nltk.tokenize.WordPunctTokenizer()
punct_tokens = punct_tokenizer.tokenize(sentence)

space_tokenizer = nltk.tokenize.SpaceTokenizer()
space_tokens = space_tokenizer.tokenize(sentence)

print("DEFAULT: ", tree_tokens)
print("PUNCT  : ", punct_tokens)
print("SPACE  : ", space_tokens)


# Parts of speech (PoS)

nltk.download('averaged_perceptron_tagger')
pos = nltk.pos_tag(tree_tokens)
print(pos)
pos_space = nltk.pos_tag(space_tokens)
print(pos_space)


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

import re
regex = re.compile("^N.*")
nouns = []
for l in pos:
    if regex.match(l[1]):
        nouns.append(l[0])
print("Nouns:", nouns)


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


