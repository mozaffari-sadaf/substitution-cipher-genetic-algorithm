# Substitution Cipher Genetic Algorithm
# This module implements a genetic algorithm to solve the substitution cipher problem.
# It includes functions for generating random DNA sequences, creating offspring, calculating fitness,
# and decoding encoded text using a substitution cipher.
# The main function demonstrates the usage of the genetic algorithm to decode a given text.

import random
import re
import string
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import requests

NUM_ITERATIONS = 1000


class SubstitutionCipher:
    def __init__(self):
        self.letter_mapping = {}  # mapping letters to their decoded ones

    def generate_mapping(self):
        """
        Generate a random mapping of letters to create substitution ciphers.

        This function substitutes letters randomly and maps them to an ordered set of letters.
        The resulting mapping is stored in the global variable `letter_mapping`.

        :return: None
        """

        letters = list(string.ascii_lowercase)
        random.shuffle(letters)
        self.letter_mapping = dict(zip(list(string.ascii_lowercase), letters))

    def encode_text(self, original_text):
        """
        Encodes the given text using the provided letter mapping(letter_mapping).

        :param str original_text: The text to be encoded.
        :return: The encoded text.
        :rtype: str
        """

        encoded_text = ''
        cleaned_text = self.clean_text(original_text.lower())

        for character in cleaned_text:
            if character in self.letter_mapping:
                encoded_text += self.letter_mapping[character]
            else:
                encoded_text += character  # handling space

        return encoded_text

    def decode_text(self, encoded_text, mapping):
        """
        Decodes the given text using the provided mapping.

        This function takes a text and a mapping and applies the mapping to each character
        in the text to decode it. Characters not found in the mapping are kept unchanged.

        :param str encoded_text: The text to be decoded.
        :param mapping: The mapping to be used for decoding.
        :return: The decoded text.
        :rtype: str
        """

        decoded_text = ''
        cleaned_text = self.clean_text(encoded_text)

        for character in cleaned_text:
            if character in mapping:
                decoded_text += mapping[character]
            else:
                decoded_text += character

        return decoded_text


    def clean_text(self, original_text):
        """
        Remove non-alpha characters and substitute them with space.

        This function takes a text and removes any characters that are not alphabetic.
        It replaces those characters with a space to maintain word separation.

        :param original_text: The given text to clean.
        :return: The cleaned text.
        :rtype: str
        """

        cleaned_text = re.compile('[^a-zA-Z]').sub(' ', original_text)
        return cleaned_text


def get_random_dna(number):
    """
    Generate random DNA sequences for the specified number of times.

    :param int number: Number of generated DNA sequences.
    :return: list of randomly generated DNA sequences.
    :rtype: list of str
    """

    population = []

    for _ in range(number):
        letters = list(string.ascii_lowercase)
        random.shuffle(letters)
        dna_sequence = letters.copy()
        population.append(dna_sequence)
    return population


def create_offspring(parent, children_count):
    """
    Generate offsprings based on the given parent for the specified number of times.

    :param list of list of str parent: A list of parent DNA sequences.
    :param int children_count: Number of generated offsprings.
    :return: The list of generated offspring and the original parent DNA sequences.
    :rtype: list of list of str
    """

    offsprings = []
    for dna in parent:
        dna_list = cipher.clean_text(dna).split()
        for _ in range(children_count):
            parent_copy = dna_list.copy()

            random_indexes = random.sample(range(len(parent_copy)), 2)
            temp_var = parent_copy[random_indexes[0]]
            parent_copy[random_indexes[0]] = parent_copy[random_indexes[1]]
            parent_copy[random_indexes[1]] = temp_var

            offsprings.append(parent_copy)
            offsprings.append(dna_list)

    return offsprings


def calculate_fitness(dna_pool, max_likelihood, best_mapping, best_dna):
    """
    Calculate the fitness of DNA sequences and find the fittest sequence.

    :param list of list of str dna_pool: A list of DNA sequence to evaluate for fitness.
    :param float max_likelihood: The current max likelihood score.
    :param dict best_mapping: The current best DNA coding map.
    :param list of str best_dna: The current best DNA sequence.
    :return: A tuple containing the best DNA sequence, its coding map, its likelihood score,
             and a dictionary of each DNA sequence associated with its likelihood.
    :rtype: tuple[list[str], dict, float, dict]
    """

    return find_best_dna(dna_pool, max_likelihood, best_mapping, best_dna)


def find_best_dna(dna_pool, max_likelihood, best_mapping, best_dna):
    """
    Find the fittest DNA sequence and its associated information.

    :param list of list of str dna_pool: A list of DNA sequence to evaluate for fitness.
    :param float max_likelihood: The current max likelihood score.
    :param dict best_mapping: The current best DNA coding map.
    :param list of str best_dna: The current best DNA sequence.
    :return: A tuple containing the best DNA sequence, its coding map, its likelihood score,
             and a dictionary of each DNA sequence associated with its likelihood.
    :rtype: tuple[list[str], dict, float, dict]
    """

    dna_score = {}
    for dna in dna_pool:
        letters = list(string.ascii_lowercase)
        dna_map = dict(zip(letters, list(dna)))
        likelihood = calculate_likelihood(encoded_text, dna_map)

        dna_score[str(dna)] = likelihood
        if likelihood > max_likelihood:
            max_likelihood = likelihood
            best_mapping = dna_map
            best_dna = dna

    return max_likelihood, best_mapping, best_dna, dna_score


def calculate_likelihood(encoded_text, dna_map):
    """
    Calculate the likelihood of a text using the provided DNA mapping.

    :param str text: The text to calculate the likelihood for.
    :param dict dna_map: The DNA mapping to be used for decoding.
    :return: The likelihood of the text.
    :rtype: float
    """

    decoded_text = cipher.decode_text(encoded_text, dna_map)
    return sentence_probability(decoded_text)


def get_train_data():
    """
    Retrieve the training data.

    :return: A string representing the content of the training data.
    :rtype: str
    """

    data_url = "https://lazyprogrammer.me/course_files/moby_dick.txt"
    try:
        response = requests.get(data_url)
        response.raise_for_status()  # Raise an exception if the response status code indicates an error
        return response.text
    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching data:", e)
        return "ERROR: Failed to fetch data"


def create_lang_model():
    """
    Create a language model based on the training data.

    This function calculates the probabilities of unigrams and bigrams based on the training data.
    It updates the global variables `uni_prob` and `bi_prob` with the normalized probabilities.

    :return: A tuple of two floats representing the normalized probabilities of unigrams and bigrams.
    """

    uni_prob = np.zeros(26)
    bi_prob = np.ones((26, 26))

    original_text = get_train_data()
    cleaned_text = cipher.clean_text(original_text)
    words = cleaned_text.lower().split()

    for w in words:
        update_uni(w[0], uni_prob)
        for character in range(len(w)-1):
            update_bi(w[character], w[character+1], bi_prob)

    bi_prob_normalized = normalize_bi(bi_prob)
    uni_prob_normalized = normalize_uni(uni_prob)

    return bi_prob_normalized, uni_prob_normalized


def normalize_bi(bi_prob):
    """
    Normalize the bi_prob matrix.

    This function takes the bi_prob matrix, which represents the probabilities of bigrams,
    and normalizes it to ensure that the probabilities sum up to 1.

    :param numpy.ndarray bi_prob: The input bi_prob matrix.
    :return: The normalized bi_prob matrix.
    :rtype: numpy.ndarray
    """

    bi_prob_normalized = bi_prob / bi_prob.sum(axis=1, keepdims=True)
    return bi_prob_normalized


def normalize_uni(uni_prob):
    """
    Normalize the given unigram probabilities.

    This function takes the unigram probabilities represented by the `uni_prob` array
    and normalizes them to ensure that the probabilities sum up to 1.

    :param numpy.ndarray uni_prob: The input array of unigram probabilities.
    :return: The normalized unigram probabilities.
    :rtype: numpy.ndarray
    """

    uni_prob_normalized = uni_prob / uni_prob.sum()
    return uni_prob_normalized


def update_uni(letter, uni_prob):
    """
    Update the unigram probabilities based on the given letter.

    This function increments the count of the specified letter in the unigram probabilities
    array (`uni_prob`). It updates the corresponding index of the letter in the array.

    :param str letter: The letter to update the unigram probabilities for.
    :param numpy.ndarray uni_prob: The array of unigram probabilities.
    :return: None
    """

    index = ord(letter) - 97
    uni_prob[index] += 1


def update_bi(letter1, letter2, bi_prob):
    """
    Update the bigram probabilities based on the given letters.

    This function increments the count of the specified letters in the bigram probabilities
    array (`bi_prob`). It updates the corresponding index of the letters in the matrix.

    :param str letter1: The first letter to update the bigram probabilities for.
    :param str letter2: The second letter to update the bigram probabilities for.
    :param numpy.ndarray bi_prob: The matrix of bigram probabilities.
    :return: None
    """

    index1 = ord(letter1) - 97
    index2 = ord(letter2) - 97

    bi_prob[index1][index2] += 1


def word_probability(word):
    """
    Calculate the probability of the given token to be a valid word.

    This function calculates the probability of the given token (`word`) to be a valid word
    based on the unigram and bigram probabilities of the language model.

    :param str word: The given token to calculate the probability for.
    :return: The probability of the token being a valid word.
    :rtype: float
    """

    first_letter = ord(word[0].lower()) - 97
    probability = np.log(uni_prob[first_letter])

    for character in range(1, len(word)):
        index1 = ord(word[character-1].lower()) - 97
        index2 = ord(word[character].lower()) - 97
        probability += np.log(bi_prob[index1][index2])

    return probability


def sentence_probability(original_text):
    """
    Calculate the probability of the given text to be valid
    based on the calculated word probabilities.

    This function calculates the probability of the given text to be a valid sentence
    by summing up the word probabilities of each individual word in the text.

    :param str text: The given text to calculate the probability for.
    :return: The probability of the text being valid.
    :rtype: float
    """

    cleaned_text = cipher.clean_text(original_text)
    words = cleaned_text.lower().split()
    probability = 0

    for w in words:
        probability += word_probability(w)

    return probability


if __name__ == '__main__':
    original_text = input("Enter the original text: ")
    cipher = SubstitutionCipher()
    cipher.generate_mapping()
    encoded_text = cipher.encode_text(original_text)

    bi_prob, uni_prob = create_lang_model()

    dna_pool = get_random_dna(20)
    scores = np.zeros(NUM_ITERATIONS)

    max_likelihood = -float('inf')
    best_mapping = None
    best_dna = None

    for i in range(NUM_ITERATIONS):
        if i > 0:   # exclude the first iteration
            # 3 offspring per parent
            dna_pool = create_offspring(dna_pool, 3)

        max_likelihood, best_mapping, best_dna, dna_score = calculate_fitness(dna_pool, max_likelihood, best_mapping, best_dna)
        scores[i] = np.mean(list(dna_score.values()))

        # sort dna by fitness and keep top 5 offsprings
        sorted_dna = sorted(dna_score.items(), key=lambda x: x[1], reverse=True)[:5]
        dna_pool = [key for key, _ in sorted_dna]

    decoded_text = cipher.decode_text(encoded_text, best_mapping)

    for k, v in cipher.letter_mapping.items():
        if best_mapping[v] != k:
            print("The letter", k, "is detected as", best_mapping[v], "wrongly")

    print("\ndecoded message:\n", textwrap.fill(decoded_text))

    plt.plot(scores)
    plt.show()