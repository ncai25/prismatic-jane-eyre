import json
from spacy.lang.en import STOP_WORDS as en_stop_words
from spacy.lang.fr import STOP_WORDS as fr_stop_words
from nltk.corpus import stopwords

en_stop_nltk = set(stopwords.words('english'))
en_stops = set(en_stop_words)
en_stops.update(en_stop_nltk)

fr_stops_nltk = set(stopwords.words('french'))
fr_stops = set(fr_stop_words)
fr_stops.update(fr_stops_nltk)

en_stops.update([
    # Chapter markers and numbers
    "chapter", "chapters", "part", "section", "page", "pages",
    'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
    'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx',
    'xxi', 'xxii', 'xxiii', 'xxiv', 'xxv', 'xxvi', 'xxvii', 'xxviii', 'xxix', 'xxx',
    'xxxi', 'xxxii', 'xxxiii', 'xxxiv', 'xxxv', 'xxxvi', 'xxxvii', 'xxxviii',
    
    # Character names from Jane Eyre
    "jane", "eyre", "rochester", "edward", "fairfax", "adele", "adèle", 
    "bertha", "mason", "helen", "burns", "diana", "mary", "rivers",
    "st", "john", "reed", "eliza", "georgiana", "bessie", "hannah",
    "grace", "poole", "blanche", "ingram", "rosamond", "oliver",
    "lloyd", "brocklehurst", "temple", "miller", "scatcherd",
    "leaven", "abbott", "briggs", "eshton", "lynn", "dent",
    "thornfield", "lowood", "gateshead", "ferndean", "moor",
    "whitcross", "morton", "millcote", "jonah","hollande"
    
    # Common function words often in literature
    "mr", "mrs", "miss", "ms", "dr", "sir", "madam", "lord", "lady",
    
    # Time markers
    "o'clock", "oclock", "morning", "evening", "night", "day", "days",
    "week", "weeks", "month", "months", "year", "years",
    "yesterday", "today", "tomorrow", "tonight",
    
    # Common verbs that might not be meaningful
    "said", "says", "saying", "say", "told", "tell", "tells", "telling",
    "asked", "ask", "asks", "asking", "answered", "answer", "answers",
    "went", "go", "goes", "going", "came", "come", "comes", "coming",
    "looked", "look", "looks", "looking", "saw", "see", "sees", "seeing",
    "thought", "think", "thinks", "thinking", "felt", "feel", "feels", "feeling",
    "made", "make", "makes", "making", "took", "take", "takes", "taking",
    "gave", "give", "gives", "giving", "got", "get", "gets", "getting",
    "put", "puts", "putting", "let", "lets", "letting",
    
    # Common adverbs and intensifiers
    "very", "quite", "rather", "really", "just", "only", "even", "still",
    "perhaps", "maybe", "probably", "certainly", "surely", "indeed",
    
    # Dialogue tags and fillers
    "oh", "ah", "eh", "well", "yes", "no", "yeah", "okay", "ok",
    
    # Pronouns variations
    "one", "ones", "something", "nothing", "everything", "anything",
    "someone", "anyone", "everyone", "nobody", "somebody", "anybody",
    
    # Common prepositions and conjunctions not caught
    "upon", "within", "without", "towards", "toward", "besides",
    "moreover", "however", "therefore", "thus", "hence", "whereas",
    
    # Other common words
    "thing", "things", "way", "ways", "time", "times", "place", "places",
    "man", "woman", "men", "women", "people", "person", "persons",
    "hand", "hands", "eye", "eyes", "face", "head", "voice",
    "room", "rooms", "house", "home", "door", "doors", "window", "windows"
])

fr_stops.update([
    # Contractions with j'
    "j'ai", "j’ai", "j'avais", "j’avais", "j'aurai", "j’aurai", "j'aurais", "j’aurais", 
    "j'étais", "j’étais", "j'aie", "j’aie", "j'en", "j’en", "j'y", "j’y",

    # Contractions with c'
    "c'est", "c’est", "c'était", "c’était", "c'étaient", "c’étaient", "c'est-à-dire", "c’est-à-dire",

    # Contractions with d'
    "d'être", "d’être", "d'avoir", "d’avoir", "d'un", "d’un", "d'une", "d’une", 
    "d'autres", "d’autres", "d'ici", "d’ici", "d'accord", "d’accord",

    # Contractions with l'
    "l'ai", "l’ai", "l'avais", "l’avais", "l'on", "l’on", "l'un", "l’un", "l'une", "l’une", 
    "l'autre", "l’autre", "l'était", "l’était", "l'avait", "l’avait",

    # Contractions with m'
    "m'a", "m’a", "m'avait", "m’avait", "m'as", "m’as", "m'en", "m’en", "m'y", "m’y", 
    "m’être", "m'etre", "m’être", "m’etre",  # both accents and plain

    # Contractions with n'
    "n'ai", "n’ai", "n'a", "n’a", "n'avait", "n’avait", "n'est", "n’est", "n'était", "n’était", 
    "n'en", "n’en", "n'y", "n’y",

    # Contractions with s'
    "s'est", "s’est", "s'était", "s’était", "s'en", "s’en", "s'y", "s’y",

    # Contractions with t'
    "t'ai", "t’ai", "t'as", "t’as", "t'a", "t’a", "t'en", "t’en", "t'y", "t’y",

    # Contractions with qu'
    "qu'il", "qu’il", "qu'elle", "qu’elle", "qu'on", "qu’on", "qu'un", "qu’un", "qu'une", "qu’une", 
    "qu'ils", "qu’ils", "qu'elles", "qu’elles", "qu'y", "qu’y", "qu'en", "qu’en", 
    "qu'est-ce", "qu’est-ce",
    
    # Character names from Jane Eyre (French versions)
    "jane", "eyre", "rochester", "edward", "édouard", "fairfax", "adèle", "adele",
    "bertha", "mason", "hélène", "helen", "burns", "diana", "marie", "mary", "rivers",
    "saint-john", "st-john", "john", "jean", "reed", "eliza", "élise", "georgiana", "georgina",
    "bessie", "hannah", "grâce", "grace", "poole", "blanche", "ingram", "rosamond", "rosemonde",
    "oliver", "lloyd", "brocklehurst", "temple", "miller", "scatcherd",
    "leaven", "abbott", "briggs", "eshton", "lynn", "dent",
    "thornfield", "lowood", "gateshead", "ferndean", "whitcross", "morton", "millcote","jonah",
    "hollande"

    # Misc compound or pronoun forms
    "peut-être", "vis-à-vis", "c'est-à-dire", "c’est-à-dire", "au-dessus", "au-dessous",
    "quelqu'un", "quelqu’un", "quelqu'une", "quelqu’une", "chacun", "chacune", 
    "aucun", "aucune", "nul", "nulle",

    # Verb forms (être, avoir)
    "suis", "es", "est", "sommes", "êtes", "sont",
    "étais", "était", "étions", "étiez", "étaient",
    "ai", "as", "a", "avons", "avez", "ont",
    "aurai", "auras", "aura", "aurons", "aurez", "auront",
    "avais", "avait", "avions", "aviez", "avaient",

    # Pronouns and determiners
    "celui-ci", "celui-là", "celle-ci", "celle-là",
    "ceux-ci", "ceux-là", "celles-ci", "celles-là",

    # Common verbs
    "dit", "dis", "disent", "dire", "disant", "disait", "disais",
    "fait", "fais", "font", "faire", "faisant", "faisait", "faisais",
    "va", "vas", "vais", "aller", "allant", "allait", "allais", "vont",
    "vient", "viens", "venir", "venant", "venait", "venais", "viennent",
    "peut", "peux", "pouvoir", "pouvant", "pouvait", "pouvais", "peuvent",
    "doit", "dois", "devoir", "devant", "devait", "devais", "doivent",
    "veut", "veux", "vouloir", "voulant", "voulait", "voulais", "veulent",
    "sait", "sais", "savoir", "sachant", "savait", "savais", "savent",

    # Time markers
    "matin", "soir", "nuit", "jour", "jours", "journée",
    "semaine", "semaines", "mois", "an", "ans", "année", "années",
    "hier", "aujourd'hui", "aujourd’hui", "demain", "maintenant", "alors", "puis",

    # Common words
    "chose", "choses", "façon", "manière", "temps", "fois",
    "homme", "femme", "hommes", "femmes", "gens", "personne", "personnes",
    "main", "mains", "œil", "yeux", "tête", "visage", "voix",
    "chambre", "maison", "porte", "portes", "fenêtre", "fenêtres",

    # Titles
    "monsieur", "madame", "mademoiselle", "messieurs", "mesdames",

    # Common adverbs and intensifiers
    "très", "assez", "plus", "moins", "beaucoup", "peu", "trop",
    "bien", "mal", "mieux", "pire", "fort", "vite", "loin", "près",
    "souvent", "toujours", "jamais", "parfois", "quelquefois",
    "encore", "déjà", "bientôt", "enfin", "aussitôt", "longtemps",
    "peut-être",

    # Dialogue fillers
    "oui", "non", "si", "bon", "ah", "oh", "eh", "hein",

    # Other
    "-NULL-", "-UNK-", "chapitre"
])


def filter_word_pairs(input_file='ibm_model/word_pairs/IBM2/word_pairs_epoch_15.json', 
                      output_file='word_pairs_filtered.json'):
    with open(input_file, 'r') as f:
        word_pairs = json.load(f)

    filtered_pairs = {}

    for en_word, stats in word_pairs.items():
        if (en_word.lower() in en_stops) or (en_word.lower() in fr_stops):
            continue

        # first pass: translations and their sentence indices
        translation_data = {}
        for fr_word, fr_stats in stats['french_translations'].items():
            if fr_word.lower() not in fr_stops:
                # store the french word with its occurrences and total count
                translation_data[fr_word] = {
                    'stats': fr_stats,
                    'sentence_indices': set(occ['sent_index'] for occ in fr_stats['occurrences'])
                }
        
        # second pass: filter out translations that appear in same sentences but have lower counts
        filtered_translations = {}
        for fr_word, data in translation_data.items():
            should_keep = True
            
            # check against other translations
            for other_fr_word, other_data in translation_data.items():
                if fr_word != other_fr_word:
                    # check if they share any sentence indices
                    shared_sentences = data['sentence_indices'] & other_data['sentence_indices']
                    
                    if shared_sentences:
                        # if they share sentences and this word has fewer total occurrences, filter it out
                        if data['stats']['total_count'] < other_data['stats']['total_count']:
                            should_keep = False
                            break
                        # if they have the same count, compare average probability
                        elif data['stats']['total_count'] == other_data['stats']['total_count']:
                            if data['stats']['avg_prob'] < other_data['stats']['avg_prob']:
                                should_keep = False
                                break
            
            if should_keep:
                filtered_translations[fr_word] = data['stats']
        
        if filtered_translations:
            filtered_pairs[en_word] = {
                'total_translation_count': len(filtered_translations),
                'french_translations': filtered_translations
            }
    
    total_en_before = len(word_pairs)
    total_en_after = len(filtered_pairs)
    print(f"\nTotal English words before filtering: {total_en_before}")
    print(f"Total English words after filtering: {total_en_after}")
    print(f"Removed {total_en_before - total_en_after} English words ({(total_en_before - total_en_after) / total_en_before * 100:.1f}%)")

    with open(output_file, 'w') as f:
        json.dump(filtered_pairs, f, indent=2, ensure_ascii=False)
        
    return filtered_pairs

if __name__ == "__main__":
    filtered_pairs = filter_word_pairs()