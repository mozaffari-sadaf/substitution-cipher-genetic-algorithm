# Substitution Cipher Genetic Algorithm
This repository contains an implementation of a substitution cipher solver using a genetic algorithm. The genetic algorithm is a heuristic search technique inspired by the process of natural selection. It aims to find the optimal solution to a problem by iteratively evolving a population of candidate solutions.

## Overview
This project aims to decrypt a given substitution cipher text without prior knowledge of the key. The algorithm utilizes a genetic algorithm approach to discover the most likely key that maps the encrypted characters to their original counterparts.

## Features
- Genetic algorithm implementation for solving substitution ciphers
- Fitness function based on frequency analysis and language models
- Random initialization and mutation operators
- Selection and crossover operators for breeding new generations
- Configurable parameters for fine-tuning the algorithm's behaviour

## Usage
1- clone the repository:
```
git clone https://github.com/mozaffari-sadaf/substitution-cipher-genetic-algorithm.git
```

2- Install the required dependencies:
```
pip install -r requirements.txt
```

3- Run the algorithm with your substitution cipher text:
```
python3 substitution_cipher.py
```

4- The program will start the genetic algorithm and output the wrong mappings along with the decrypted text.

## Example

Here's an example of how to use the algorithm with a substitution cipher text:
Input:
```
I then lounged down the street and found, as I expected, that there was a mews in a lane which runs down by one wall of the garden. I lent the ostlers a hand in rubbing down their horses, and received in exchange twopence, a glass of half-and-half, two fills of shag tobacco, and as much information as I could desire about Miss Adler, to say nothing of half a dozen other people in the neighbourhood in whom I was not in the least interested, but whose biographies I was compelled to listen to.
```

Output:
```
The letter k is detected as z wrongly
The letter z is detected as k wrongly

decoded message:
 i then lounged down the street and found  as i expected  that there
was a mews in a lane which runs down by one wall of the garden  i lent
the ostlers a hand in rubbing down their horses  and received in
exchange twopence  a glass of half and half  two fills of shag tobacco
and as much information as i could desire about miss adler  to say
nothing of half a doken other people in the neighbourhood in whom i
was not in the least interested  but whose biographies i was compelled
to listen to
```
## Contributing
Contributions are welcome!
