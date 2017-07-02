import argparse
import hashlib
import itertools
import os
import random
import time

import cv2
import numpy as np
import re
import yaml

def extract_words(text, word_count):
    """
    Extract a list of words from a text in sequential order.
    :param text: source text, tokenized
    :param word_count: number of words to return
    :return: list of words
    """
    text_length = len(text)

    if word_count > text_length:
        raise RuntimeError('Cannot extract {} words from a text of {} words.'.format(word_count, text_length))

    # Determine start index
    max_range = text_length - word_count
    start_range = random.randrange(max_range)

    return text[start_range:start_range + word_count]

def get_character_image(character, glyph_source, glyph_family):
    """
    Get a random character image.
    :param character: character to obtain an image for
    :param glyph_source: directory where glyphs are located
    :return: np.array image of glyph in grayscale
    """
    glyph_file = get_glyph_file(character, glyph_source, glyph_family)
    glyph_image = cv2.imread(glyph_file, cv2.IMREAD_GRAYSCALE)

    if glyph_image is None:
        raise FileNotFoundError(glyph_file)

    return glyph_image

def get_glyph_file(glyph, glyph_source, glyph_family = None):
    directory = str(ord(glyph))
    # if directory in get_glyph_file.cache:
    # Directory in cache variable, pick any
    if glyph_family:
        glyph_file = glyph_family
    else:
        glyph_file = random.choice(list(get_glyph_file.cache[directory]))

    glyph_file = os.path.join(glyph_source, directory, glyph_file)
    return glyph_file
    # else:
    #     # Load directory
    #     get_glyph_file.cache[directory] = {}
    #     regex = re.compile(r'[^.]+\.(jpg|png)$')
    #     glyph_directory = os.path.join(glyph_source, directory)
    #     files = os.listdir(glyph_directory)
    #     glyphs = list(filter(lambda x: regex.match(x), files))
    #
    #     for glyph in glyphs:
    #         get_glyph_file.cache[directory][glyph] = None

get_glyph_file.cache = {}

def initialize_glyph_cache(glyph_set, glyph_source):
    regex = re.compile(r'[^.]+\.(jpg|png)$')
    for glyph_index, glyph in enumerate(glyph_set):
        # Load directory
        directory = str(glyph)
        get_glyph_file.cache[directory] = {}
        glyph_directory = os.path.join(glyph_source, directory)
        files = os.listdir(glyph_directory)
        glyphs = list(filter(lambda x: regex.match(x), files))

        for glyph in glyphs:
            get_glyph_file.cache[directory][glyph] = None

def get_glyph_family():
    """
    Return the identifier used by a glyph family
    :return:
    """
    # for glyph_index, glyph in enumerate(glyph_set):
        # get_glyph_file(glyph, glyph_source)

    glyph_dictionary = random.choice(list(get_glyph_file.cache))
    glyph_family = random.choice(list(get_glyph_file.cache[glyph_dictionary]))
    return glyph_family

def estimate_word_width(word, glyph_set, glyph_width_probabilities):
    """
    Given a word, estimate its width given the glyph average width.
    :param word: word to estimate the width
    :param glyph_set: list supported glyphs
    :param glyph_width_probabilities: glyph width (average width + number of samples)
    :return: int the expected word width
    """
    width = 0

    glyph_map = {}
    for glyph_index, glyph in enumerate(glyph_set):
        glyph_map[chr(glyph)] = glyph_index

    for glyph in word:
        index = glyph_map[glyph]
        glyph_width = glyph_width_probabilities[index, 0]
        width += glyph_width

    return int(width)

def trim_padding(img, value = 255):
    """
    Trim rows and columns containing the specified value.
    :param img:  2D digit array
    :param value: which value should be trimmed
    :return: np.array trimmed digit
    """
    # mask_row = np.all(np.equal(img, value), axis=1)
    # dst = img[~mask_row]

    dst = img.copy()

    mask_col = np.all(np.equal(dst, value), axis=0)
    dst = dst[:, ~mask_col]

    if dst.shape[0] == 0 or dst.shape[1] == 0:
        return img

    return dst

def synthesize(args):
    """
    Synthesize a collection of images.
    :param args: command line arguments
    :return: None
    """
    count = args.count
    text_source = args.text_source
    glyph_source = args.glyph_source
    target_directory = args.target_directory
    space_width = 50
    config = None
    with open(args.config) as file:
        config = yaml.load(file.read())

    glyph_set = range(33, 127) # ! to ~

    # Create target directory if it does not exist
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)

    # Initialize glyph cache
    initialize_glyph_cache(glyph_set, glyph_source)

    text = []
    with open(text_source) as file:
        text = file.read().split()

    # Filter out words that contain characters we do not know
    def known_words(word):
        for character in word:
            if ord(character) not in glyph_set:
                return False
        return True
    text = list(filter(known_words, text))

    for i in range(count):
        start_time = time.time()

        print('{}/{}'.format(i, count))

        # Generate a sample of words to stamp
        if args.min_words == args.max_words:
            word_count = args.min_words
        else:
            word_count = random.randrange(args.min_words, args.max_words)
        words = extract_words(text, word_count)

        # Randomly select a background
        background_config = random.choice(config['backgrounds'])
        background = background_config['file']

        image = cv2.imread(background)

        if image is None:
            raise FileNotFoundError(background)

        # Randomly select a glyph family
        glyph_family = get_glyph_family()

        formatted_text = ''
        left_margin = background_config['regions'][0][0]
        x_start = left_margin
        y_start = background_config['regions'][0][1] + 25
        line_height = background_config['line_height']
        number_of_words_on_line = 0
        yml_data = []
        for word_index, word in enumerate(words):
            # Generate glyphs for the word and determine word width
            word_data = {
                'text': word,
                'glyphs': [],
            }
            word_width = 0
            glyphs = []
            character_length = len(word)
            for character_index, character in enumerate(word):
                glyph_image_x = get_character_image(character, glyph_source, glyph_family)

                # Apply thinning and blurring
                if background_config['dilate'] > 0:
                    kernel = np.ones((background_config['dilate'], background_config['dilate']), np.uint8)
                    glyph_image_x = cv2.dilate(glyph_image_x, kernel)
                if background_config['blur'] > 0:
                    glyph_image_x = cv2.blur(glyph_image_x, (background_config['blur'], background_config['blur']))

                glyph_image = trim_padding(glyph_image_x)
                glyph_image = cv2.cvtColor(glyph_image, cv2.COLOR_GRAY2BGR)
                # TODO(tom.rochette@coreteks.org): Simulate colored pen
                # TODO(tom.rochette@coreteks.org): Apply other transforms (scale, rotation, skew, shear, elastic transform)

                glyphs.append([word_width, character, glyph_image])
                word_width += glyph_image.shape[1]
                # Random spacing between glyphs
                if character_index + 1 != character_length:
                    word_width += random.randrange(-5, 10)

            # Determine if the word fits on this line
            if x_start + word_width > background_config['regions'][1][0]:
                # Word doesn't fit, move to new line
                x_start = left_margin
                y_start += line_height
                number_of_words_on_line = 0
                formatted_text += '\n'

            if y_start + line_height >= background_config['regions'][1][1]:
                print('Could only write {} words out of {} requested.'.format(word_index + 1, word_count))
                break

            # Separate words by a space
            if number_of_words_on_line > 0:
                formatted_text += ' '

            formatted_text += word
            number_of_words_on_line += 1

            # Stamp glyph
            for x, glyph, glyph_image in glyphs:
                # As x can be negative, we don't want to go lower than the left_margin for the first word on a new line
                x_min = max(left_margin, x_start + x)
                x_max = x_min + glyph_image.shape[1]
                y_min = y_start
                y_max = y_min + glyph_image.shape[0]

                image_roi = image[y_min:y_max, x_min:x_max].astype(np.uint16)
                glyph_image_roi = glyph_image.astype(np.uint16)

                # We multiply the glyph image with the image at the stamping location, which has the effect of keeping
                # the image as is (when the glyph image pixel is 255) or replacing it with the glyph (when the glyph pixel
                # is 0). For intermediate values (1-254) we get a blend between the two, which might not be what we want.
                image[y_min:y_max, x_min:x_max] = ((image_roi*glyph_image_roi)/255).astype(np.uint8)

                word_data['glyphs'].append({
                    'glyph': glyph,
                    'x': x_min,
                    'y': y_min,
                    'width': glyph_image.shape[1],
                    'height': glyph_image.shape[0],
                })

            x_start += word_width
            x_start += space_width
            # Random spacing between words
            x_start += random.randrange(-10, 10)

            yml_data.append(word_data)

        # Compute sha1 of the image to prevent multiple with the same string but different content from being overwritten
        sha1 = hashlib.sha1()
        sha1.update(image.data)
        cv2.imwrite(os.path.join(target_directory, sha1.hexdigest() + '.jpg'), image)

        if args.txt:
            with open(os.path.join(target_directory, sha1.hexdigest() + '.txt'), 'w') as f:
                f.write(formatted_text)

        if args.yml:
            with open(os.path.join(target_directory, sha1.hexdigest() + '.yml'), 'w') as f:
                yml  = yaml.dump(yml_data)
                f.write(yml)

        elapsed_time = time.time() - start_time
        print('Elapsed: {}s'.format(elapsed_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--count', default=100, help='Number of samples to generate (default: 100)')
    parser.add_argument('--min_words', default=1, type=int, help='Minimum number of words to generate (default: 1)')
    parser.add_argument('--max_words', default=300, type=int, help='Maximum number of words to generate (default: 300)')
    parser.add_argument('--text_source', default='source.txt', help='Text file to use as source of content (default: source.txt)')
    parser.add_argument('--glyph_source', default='glyphs', help='Directory containing glyphs to stamp onto the generated images (default: glyphs)')
    parser.add_argument('--target_directory', default='generated', help='Directory where to store generated images (default: generated)')
    parser.add_argument('--txt', action='store_true', help='Generate the associated text in a txt file (default: False)')
    parser.add_argument('--yml', action='store_true', help='Generate the associated data (glyph + bounding box (top left x, y + width/height) in a yml file (default: False)')
    parser.add_argument('--config', default='config.yml', help='Configuration file containing the background details (default: config.yml)')

    args = parser.parse_args()

    synthesize(args)