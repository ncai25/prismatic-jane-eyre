#!/usr/bin/env python3
"""
Context Retrieval Script for IBM Translation Models

Usage: retrieve <sentence_index> <french_position> <english_position> <window>
Example: retrieve 58325 13 11 4
"""

import sys
import os
import linecache
from pathlib import Path

os.chdir(os.path.dirname(os.path.abspath(__file__)))

JEAN_CHAPTERS = [
    (1, 126, 1), (127, 316, 2), (317, 532, 3), (533, 901, 4), (902, 1230, 5),
    (1231, 1420, 6), (1421, 1603, 7), (1604, 1781, 8), (1782, 2016, 9), (2017, 2331, 10),
    (2332, 2741, 11), (2742, 2987, 12), (2988, 3313, 13), (3314, 3646, 14), (3647, 4001, 15),
    (4002, 4255, 16), (4256, 4821, 17), (4822, 5169, 18), (5170, 5520, 19), (5521, 6007, 20),
    (6008, 6647, 21), (6648, 6818, 22), (6819, 7115, 23), (7116, 7627, 24), (7628, 8030, 25),
    (8031, 8351, 26), (8352, 9182, 27), (9183, 9733, 28), (9734, 10057, 29), (10058, 10281, 30),
    (10282, 10492, 31), (10493, 10801, 32), (10802, 11202, 33), (11203, 11844, 34), (11845, 12167, 35),
    (12168, 12467, 36), (12468, 13136, 37), (13137, 13244, 38),
]

MAURAT_CHAPTERS = [
    (13245, 13352, 1), (13353, 13526, 2), (13527, 13740, 3), (13741, 14086, 4), (14087, 14401, 5),
    (14402, 14580, 6), (14581, 14759, 7), (14760, 14937, 8), (14938, 15143, 9), (15144, 15436, 10),
    (15437, 15831, 11), (15832, 16072, 12), (16073, 16389, 13), (16390, 16704, 14), (16705, 17028, 15),
    (17029, 17267, 16), (17268, 17792, 17), (17793, 18132, 18), (18133, 18465, 19), (18466, 18905, 20),
    (18906, 19508, 21), (19509, 19671, 22), (19672, 19949, 23), (19950, 20441, 24), (20442, 20820, 25),
    (20821, 21101, 26), (21102, 21856, 27), (21857, 22356, 28), (22357, 22664, 29), (22665, 22877, 30),
    (22878, 23075, 31), (23076, 23362, 32), (23363, 23729, 33), (23730, 24324, 34), (24325, 24628, 35),
    (24629, 24906, 36), (24907, 25516, 37), (25517, 25622, 38),
]

MONOD_CHAPTERS = [
    (25623, 25746, 1), (25747, 25940, 2), (25941, 26157, 3), (26158, 26537, 4), (26538, 26877, 5),
    (26878, 27081, 6), (27082, 27273, 7), (27274, 27466, 8), (27467, 27705, 9), (27706, 28016, 10),
    (28017, 28453, 11), (28454, 28714, 12), (28715, 29052, 13), (29053, 29416, 14), (29417, 29788, 15),
    (29789, 30065, 16), (30066, 30667, 17), (30668, 31030, 18), (31031, 31386, 19), (31387, 31886, 20),
    (31887, 32550, 21), (32551, 32731, 22), (32732, 33050, 23), (33051, 33603, 24), (33604, 34033, 25),
    (34034, 34370, 26), (34371, 35240, 27), (35241, 35829, 28), (35830, 36185, 29), (36186, 36437, 30),
    (36438, 36661, 31), (36662, 36988, 32), (36989, 37405, 33), (37406, 38081, 34), (38082, 38414, 35),
    (38415, 38725, 36), (38726, 39428, 37), (39429, 39541, 38),
]

REDON_DULONG_CHAPTERS = [
    (39542, 39642, 1), (39643, 39811, 2), (39812, 40002, 3), (40003, 40311, 4), (40312, 40598, 5),
    (40599, 40750, 6), (40751, 40902, 7), (40903, 41043, 8), (41044, 41216, 9), (41217, 41370, 10),
    (41371, 41705, 11), (41706, 41862, 12), (41863, 42122, 13), (42123, 42367, 14), (42368, 42628, 15),
    (42629, 42843, 16), (42844, 43229, 17), (43230, 43463, 18), (43464, 43768, 19), (43769, 44172, 20),
    (44173, 44749, 21), (44750, 44889, 22), (44890, 45152, 23), (45153, 45596, 24), (45597, 45939, 25),
    (45940, 46234, 26), (46235, 46949, 27), (46950, 47355, 28), (47356, 47656, 29), (47657, 47862, 30),
    (47863, 48032, 31), (48033, 48287, 32), (48288, 48597, 33), (48598, 49131, 34), (49132, 49398, 35),
    (49399, 49599, 36), (49600, 50077, 37), (50078, 50174, 38),
]

SOUVESTRE_CHAPTERS = [
    (50175, 50274, 1), (50275, 50450, 2), (50451, 50660, 3), (50661, 51017, 4), (51018, 51347, 5),
    (51348, 51545, 6), (51546, 51725, 7), (51726, 51910, 8), (51911, 52126, 9), (52127, 52433, 10),
    (52434, 52848, 11), (52849, 53086, 12), (53087, 53400, 13), (53401, 53714, 14), (53715, 54031, 15),
    (54032, 54271, 16), (54272, 54806, 17), (54807, 55141, 18), (55142, 55467, 19), (55468, 55923, 20),
    (55924, 56518, 21), (56519, 56675, 22), (56676, 56960, 23), (56961, 57463, 24), (57464, 57855, 25),
    (57856, 58152, 26), (58153, 58928, 27), (58929, 59463, 28), (59464, 59778, 29), (59779, 60006, 30),
    (60007, 60210, 31), (60211, 60504, 32), (60505, 60885, 33), (60886, 61515, 34), (61516, 61835, 35),
    (61836, 62124, 36), (62125, 62779, 37), (62780, 62887, 38),
]

def get_translation_version(sentence_index):
    """Get translation version based on sentence index."""
    if 1 <= sentence_index <= 13244:
        return "JEAN"
    elif 13245 <= sentence_index <= 25622:
        return "MAURAT"
    elif 25623 <= sentence_index <= 39541:
        return "MONOD"
    elif 39542 <= sentence_index <= 50174:
        return "REDON_DULONG"
    elif 50175 <= sentence_index <= 62887:
        return "SOUVESTRE"
    else:
        return "Unknown"

def get_chapter_info(sentence_index):
    """Get chapter number and translation version for a given sentence index."""
    version = get_translation_version(sentence_index)
    
    if version == "JEAN":
        chapter_list = JEAN_CHAPTERS
    elif version == "MAURAT":
        chapter_list = MAURAT_CHAPTERS
    elif version == "MONOD":
        chapter_list = MONOD_CHAPTERS
    elif version == "REDON_DULONG":
        chapter_list = REDON_DULONG_CHAPTERS
    elif version == "SOUVESTRE":
        chapter_list = SOUVESTRE_CHAPTERS
    else:
        return None, version
    
    for start, end, chapter in chapter_list:
        if start <= sentence_index <= end:
            return chapter, version
    
    return None, version

def get_words_with_context(path, sent_idx, word_pos, window):
    """Get words with context from a file."""
    all_words = []
    sentence_boundaries = []
    
    if sent_idx > 0 and word_pos < window:
        prev = linecache.getline(path, sent_idx).strip().split()
        all_words.extend(prev)
        sentence_boundaries.append(len(prev))
    
    curr = linecache.getline(path, sent_idx + 1).strip().split()
    sentence_start = len(all_words)
    all_words.extend(curr)
    
    if word_pos + window >= len(curr):
        next_sent = linecache.getline(path, sent_idx + 2).strip().split()
        all_words.extend(next_sent)
    
    actual_pos = sentence_start + word_pos
    
    start = max(0, actual_pos - window)
    end = min(len(all_words), actual_pos + window + 1)
    
    return ' '.join(all_words[start:end])

def get_bilateral_context(french_path, english_path, sent_index, french_pos, english_pos, window=20):
    """Get bilateral context for French and English."""
    fr_context = get_words_with_context(french_path, sent_index, french_pos, window)
    en_context = get_words_with_context(english_path, sent_index, english_pos, window)
    
    chapter, version = get_chapter_info(sent_index)
    
    return {
        'french_context': fr_context,
        'english_context': en_context,
        'sentence_index': sent_index,
        'chapter': chapter,
        'translation_version': version
    }

def main():
    if len(sys.argv) != 5:
        print("Usage: retrieve <sentence_index> <french_position> <english_position> <window>")
        print("Example: retrieve 58325 13 11 4")
        sys.exit(1)
    
    try:
        sent_index = int(sys.argv[1])
        french_pos = int(sys.argv[2])
        english_pos = int(sys.argv[3])
        window = int(sys.argv[4])
    except ValueError:
        print("Error: All arguments must be integers")
        sys.exit(1)
    
    french_path = "preprocess/align_tightened/fr_full.txt"
    english_path = "preprocess/align_tightened/en_full.txt"
    
    if not Path(french_path).exists():
        print(f"Error: {french_path} not found")
        sys.exit(1)
    if not Path(english_path).exists():
        print(f"Error: {english_path} not found")
        sys.exit(1)
    
    result = get_bilateral_context(french_path, english_path, sent_index, french_pos, english_pos, window)
    
    print(f"Sentence Index: {result['sentence_index']}")
    print(f"Translation: {result['translation_version']}")
    print(f"Chapter: {result['chapter']}")
    print(f"French Context: {result['french_context']}")
    print(f"English Context: {result['english_context']}")

if __name__ == "__main__":
    main()