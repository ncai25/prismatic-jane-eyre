import matplotlib.pyplot as plt
from itertools import accumulate
from collections import defaultdict
import numpy as np
import json
import os

def write_list(aer, path):
	f = open(path, 'w')
	for rate in aer:
		f.write(str(rate) + '\n')
	f.close()

def read_list(path):
	f = open(path, 'r')
	return [float(line.split('\n')[0]) for line in f]

def plot_aer(aers, path):
	plt.plot(range(len(aers)), aers)
	plt.savefig(path + 'AER-plot.pdf')
	plt.clf()


def plot_jump(jump, max_jump, path):
	xs = list(map(lambda x : x - max_jump, list(range(len(jump[0])))))
	plt.plot(xs, jump[0])
	plt.savefig(path + 'jump-plot.pdf')
	plt.clf()

def draw_alignment_null(naacl_path, french_path, english_path, fig_path, sure=False, sentence=1):
	"""

	"""
	i = 1
	f = open(french_path, 'r')
	while i <= sentence:
		french = f.readline().split()
		i += 1
	f.close()
	
	i = 1
	e = open(english_path, 'r')
	while i <= sentence:
		english = e.readline().split()
		i += 1
	english = ['NULL']+english
	e.close()

	n = open(naacl_path, 'r')
	alignments = defaultdict(list)
	line = n.readline()
	while line[0] != str(sentence):
		line = n.readline().split()
	while line[0] != str(sentence+1):
		if sure:
			if line[-1] != 'P':
				alignments[int(line[2])-1].append(int(line[1]))
		else:
			alignments[int(line[2])-1].append(int(line[1]))
		line = n.readline().split()
	
	bitext = ([(french, english)], alignments)

	fig = plt.figure(figsize=(8, 4))
	ax = plt.axes()
	plt.axis('off')
	coordinates = get_coordinates(bitext)

	
	lines = [ax.plot(xy[0], xy[1],alpha=0.9,linestyle='-',color='#1a75ff',lw=1.0, solid_capstyle='round')[0] for xy in coordinates['edges']]

	ax.scatter(coordinates['x_f']+coordinates['x_e'], coordinates['y_f']+coordinates['y_e'],
					s=30, c='white', marker='o', lw=0, alpha=1)

	plot_words(ax, coordinates['x_f'], coordinates['y_f'], coordinates['w_f'], 'top')
	plot_words(ax, coordinates['x_e'], coordinates['y_e'], coordinates['w_e'], 'bottom')
	
	plt.savefig(fig_path)

def draw_all_alignments(french_path, english_path, fig_path, sure=False, sentence=1):
	"""
	Draws an alignment that is weighted according to probs of alignment.
	"""
	i = 1
	f = open(french_path, 'r')
	while i <= sentence:
		french = f.readline().split()
		i += 1
	f.close()
	
	i = 1
	e = open(english_path, 'r')
	while i <= sentence:
		english = e.readline().split()
		i += 1
	english = ['NULL']+english
	e.close()

	alignments = defaultdict(list)
	
	bitext = ([(french, english)], alignments)

	fig = plt.figure(figsize=(8, 4))
	ax = plt.axes()
	plt.axis('off')
	coordinates = get_coordinates(bitext, draw_all=True)

	raw_line_weights = []
	
	for j, f in enumerate(french):
	# for i, e in enumerate(english):
		for i, e in enumerate(english):
		# for j, f in enumerate(french):
			raw_line_weights.append(1./len(english))

	# print(len(raw_line_weights))
	line_weights = [w*10 for w in raw_line_weights]
	print(line_weights)
	edge_coords = coordinates['edges']
	print(len(edge_coords))
	lines = [ax.plot(xy[0], xy[1], alpha=0.9, linewidth=w, linestyle='-', color='#1a75ff', solid_capstyle='round')[0] for xy,w in zip(coordinates['edges'], line_weights)]


	ax.scatter(coordinates['x_f']+coordinates['x_e'], coordinates['y_f']+coordinates['y_e'],
					s=30, c='white', marker='o', lw=0, alpha=1)

	plot_words(ax, coordinates['x_f'], coordinates['y_f'], coordinates['w_f'], 'top')
	plot_words(ax, coordinates['x_e'], coordinates['y_e'], coordinates['w_e'], 'bottom')
	
	plt.savefig(fig_path + '.pdf')
	plt.clf()

def draw_weighted_alignment_null(model, naacl_path, french_path, english_path, fig_path, sure=False, sentence=1, all_ones=False):
	"""
	Draws an alignment that is weighted according to probs of alignment.
	"""
	i = 1
	f = open(french_path, 'r')
	while i <= sentence:
		french = f.readline().split()
		i += 1
	f.close()
	
	i = 1
	e = open(english_path, 'r')
	while i <= sentence:
		english = e.readline().split()
		i += 1
	english = ['NULL']+english
	e.close()

	alignments = defaultdict(list)

	bitext = ([(french, english)], alignments)

	fig = plt.figure(figsize=(8, 4))
	ax = plt.axes()
	plt.axis('off')
	coordinates = get_coordinates(bitext, draw_all=True)

	# get weights of the edges:

	F_indices = []
	for i in french:
		try:
			f = model.V_f_indices[i]
		except KeyError:
			f = model.V_f_indices['-UNK-']
		F_indices.append(f)

	E_indices = []
	for i in english:
		try:
			e = model.V_e_indices[i]
		except KeyError:
			e = model.V_e_indices['-UNK-']
		E_indices.append(e)

	raw_line_weights = []
	
	for j, f in enumerate(F_indices):
	# for i, e in enumerate(E_indices):
		posterior = model.posterior(f, j, E_indices, F_indices)
		for i, e in enumerate(E_indices):
		# for j, f in enumerate(F_indices):
			
			p = posterior[i]
			raw_line_weights.append(p)

	print(len(raw_line_weights))
	line_weights = [w*10 for w in raw_line_weights]
	print(line_weights)
	edge_coords = coordinates['edges']
	print(len(edge_coords))
	lines = [ax.plot(xy[0], xy[1], alpha=0.9, linewidth=w, linestyle='-', color='#1a75ff', solid_capstyle='round')[0] for xy,w in zip(coordinates['edges'], line_weights)]


	ax.scatter(coordinates['x_f']+coordinates['x_e'], coordinates['y_f']+coordinates['y_e'],
					s=30, c='white', marker='o', lw=0, alpha=1)

	plot_words(ax, coordinates['x_f'], coordinates['y_f'], coordinates['w_f'], 'top')
	plot_words(ax, coordinates['x_e'], coordinates['y_e'], coordinates['w_e'], 'bottom')
	
	plt.savefig(fig_path + '.pdf')
	plt.clf()

def draw_alignment(naacl_path, french_path, english_path, fig_path, sure=False, sentence=1):
	"""

	"""
	i = 1
	f = open(french_path, 'r')
	while i <= sentence:
		french = f.readline().split()
		i += 1
	f.close()
	
	i = 1
	e = open(english_path, 'r')
	while i <= sentence:
		english = e.readline().split()
		i += 1
	e.close()

	n = open(naacl_path, 'r')
	alignments = defaultdict(list)
	line = n.readline()
	while line[0] != str(sentence):
		line = n.readline().split()
	while line[0] != str(sentence+1):
		if sure:
			if line[-1] != 'P':
				alignments[int(line[2])-1].append(int(line[1])-1)
		else:
			alignments[int(line[2])-1].append(int(line[1])-1)
		line = n.readline().split()
	
	bitext = ([(french, english)], alignments)

	fig = plt.figure(figsize=(8, 4))
	ax = plt.axes()
	plt.axis('off')
	coordinates = get_coordinates(bitext)

	
	lines = [ax.plot(xy[0], xy[1],alpha=0.9,linestyle='-',color='#1a75ff',linewidth=1.0, solid_capstyle='round')[0] for xy in coordinates['edges']]

	ax.scatter(coordinates['x_f']+coordinates['x_e'], coordinates['y_f']+coordinates['y_e'],
					s=30, c='white', marker='o', lw=0, alpha=1)

	plot_words(ax, coordinates['x_f'], coordinates['y_f'], coordinates['w_f'], 'top')
	plot_words(ax, coordinates['x_e'], coordinates['y_e'], coordinates['w_e'], 'bottom')
	
	plt.savefig(fig_path + '.pdf')



def draw_weighted_alignment(model, naacl_path, french_path, english_path, fig_path, sure=False, sentence=1):
	"""
	Draws an alignment that is weighted according to probs of alignment.
	"""
	i = 1
	f = open(french_path, 'r')
	while i <= sentence:
		french = f.readline().split()
		i += 1
	f.close()
	
	i = 1
	e = open(english_path, 'r')
	while i <= sentence:
		english = e.readline().split()
		i += 1
	e.close()

	n = open(naacl_path, 'r')
	alignments = defaultdict(list)
	line = n.readline()
	while line[0] != str(sentence):
		line = n.readline().split()
	while line[0] != str(sentence+1):
		if sure:
			if line[-1] != 'P':
				alignments[int(line[2])-1].append(int(line[1])-1)
		else:
			alignments[int(line[2])-1].append(int(line[1])-1)
		line = n.readline().split()
	
	bitext = ([(french, english)], alignments)

	fig = plt.figure(figsize=(8, 4))
	ax = plt.axes()
	plt.axis('off')
	coordinates = get_coordinates(bitext)

	# get weights of the edges:

	# F_indices = [model.V_f_indices[i] for i in french]
	# E_indices = [model.V_e_indices[i] for i in english]
	F_indices = []
	for i in french:
		try:
			f = model.V_f_indices[i]
		except KeyError:
			f = model.V_f_indices['-UNK-']
		F_indices.append(f)

	E_indices = []
	for i in english:
		try:
			e = model.V_e_indices[i]
		except KeyError:
			e = model.V_e_indices['-UNK-']
		F_indices.append(e)
	
	raw_line_weights = []
	
	for i, e in enumerate(E_indices):
		for f in F_indices:
			try:
				p = model.posterior(f, E_indices)[i]
			except IndexError:
				p = 0
			raw_line_weights.append(p)

	line_weights = [w*10 for w in raw_line_weights]
	edge_coords = coordinates['edges']
	lines = [ax.plot(xy[0], xy[1], alpha=0.9, linewidth=w, linestyle='-', color='#1a75ff', solid_capstyle='round')[0] for xy,w in zip(coordinates['edges'], line_weights)]


	ax.scatter(coordinates['x_f']+coordinates['x_e'], coordinates['y_f']+coordinates['y_e'],
					s=30, c='white', marker='o', lw=0, alpha=1)

	plot_words(ax, coordinates['x_f'], coordinates['y_f'], coordinates['w_f'], 'top')
	plot_words(ax, coordinates['x_e'], coordinates['y_e'], coordinates['w_e'], 'bottom')
	
	plt.savefig(fig_path)
	plt.clf()


def plot_words(axes, xs, ys, words, vertical_position, weight='bold', color='black'):
	"""
	Code borrowed from: https://github.com/INFR11133/lab1

	Plots a pair of sentences at given positions, on a given axes object.
	
	Args:
	   axes (pyplotlib axes object): axes on which to plot
	   xs (list of floats): x coordinates
	   ys (list of floats): y coordinates
	   words (list of strings): words to be displayed
	   vertical_position (string): where words should be displayed relative to point coordinates
	   weight (string): font weight
	   color (string or list of strings): color/s to be used for displaying words
	"""
	for n in range(0, len(words)):
		axes.text(xs[n], ys[n], words[n], size=9, family='sans-serif',
				  weight=weight, color=color,
				  horizontalalignment='center',
				  verticalalignment=vertical_position)


def get_coordinates(bitext, draw_all=False, one_sent=False, sent_index=0, word_index=0):
	"""
	Code borrowed from: https://github.com/INFR11133/lab1

	Generates x and y coordinates to be used for plotting sentence pairs
	and alignment links.
	
	Args:
	   bitext (list of tuples): list of translation pairs
	   one_sent (Boolean): whether coordinates ahould be generated for one
		   selected sentence pair and one word in it, or for the whole bitext
	   sent_index (int): index of the selected sentence pair
	   word_index (int): index of the target foreign word
	"""
	x_positions_f = []
	y_positions_f = []
	x_positions_e = []
	y_positions_e = []
	edge_pos = []
	words_f = []
	words_e = []
	sents, alignments = bitext
	for (n, (f, e)) in enumerate(sents):

		for j in range(0, len(f)):
			x_positions_f.append(j+1)
			y_positions_f.append((3*n)-2)
			words_f.append(f[j])
			if (not one_sent) or (one_sent and word_index==j):
				for i in range(0, len(e)):
					if draw_all:
						edge_pos.append([[j+1, i+1], [(3*n)-1.9, (3*n)-1.1]])
					else:
						if i in alignments[j]:
							edge_pos.append([[j+1, i+1], [(3*n)-1.9, (3*n)-1.1]])

		for i in range(0, len(e)):
			x_positions_e.append(i+1)
			y_positions_e.append((3*n)-1)
			words_e.append(e[i])
	coord_dict = {'x_f': x_positions_f, 'x_e': x_positions_e,
			'y_f': y_positions_f, 'y_e': y_positions_e,
			'edges': edge_pos, 'w_f': words_f, 'w_e': words_e}
	return coord_dict


def save_word_pairs(ibm_model, word_pair_path, epoch_num):
	"""
	Extract and save word pairs from IBM model with normalized structure.
	
	Args:
		ibm_model: IBM1 or IBM2 model instance
		word_pair_path: Directory path to save word pairs
		epoch_num: Current epoch number
	"""
	os.makedirs(word_pair_path, exist_ok=True)
	word_pair_path_json = word_pair_path + f'word_pairs_epoch_{epoch_num}.json'
	
	def normalize_word(word):
		word = word.lower().strip('-!?.,;:""\'_★…')
		if '--' in word:
			word = word.split('--')[0].strip('!?.,;:""\'_★…')
		return word
	
	all_word_pairs = []
	for sent_idx, (F, E) in enumerate(zip(ibm_model.french, ibm_model.english)):
		# Check if this is IBM1 (has return_pairs parameter) or IBM2
		if 'IBM1' in ibm_model.__class__.__name__:
			alignment, pairs = ibm_model.align(F, E, return_pairs=True)
		else:
			# IBM2 - create pairs manually
			alignment = ibm_model.align(F, E)
			pairs = [(F[idx], E[alignment[idx]]) for idx in range(len(F))]
		
		for word_idx, (f_word, e_word) in enumerate(pairs):
			f_index = ibm_model.V_f_indices.get(f_word, ibm_model.V_f_indices.get("-UNK-", 0))
			e_index = ibm_model.V_e_indices.get(e_word, ibm_model.V_e_indices.get("-UNK-", 0))
			prob = ibm_model.t[f_index, e_index]
			
			# only keep word pairs with sufficient probability
			if prob > 0.0001:
				fr_norm = normalize_word(f_word)
				en_norm = normalize_word(e_word)
				
				if not fr_norm or not en_norm:
					continue
				
				# adjust English position for NULL word
				eng_pos = int(alignment[word_idx])
				if ibm_model.null and eng_pos > 0:
					eng_pos -= 1
				
				all_word_pairs.append({
					'french_word': f_word,
					'english_word': e_word,
					'french_normalized': fr_norm,
					'english_normalized': en_norm,
					'prob': prob,
					'sent_index': sent_idx,
					'french_pos': word_idx,
					'english_pos': eng_pos
				})
	
	# group word pairs by normalized English word, then by French word
	english_groups = defaultdict(lambda: defaultdict(list))
	
	for pair in all_word_pairs:
		eng_norm = pair['english_normalized']
		fr_norm = pair['french_normalized']
		
		occurrence = {
			'french_form': pair['french_word'],
			'english_form': pair['english_word'],
			'sent_index': pair['sent_index'],
			'french_pos': pair['french_pos'],
			'english_pos': pair['english_pos'],
			'prob': round(pair['prob'], 4)
		}
		
		english_groups[eng_norm][fr_norm].append(occurrence)
	
	# build final JSON structure with sorted French translations
	final_structure = {}
	for eng_word, french_translations in english_groups.items():
		
		french_with_avg_prob = []
		for fr_word, occurrences in french_translations.items():
			probs = [occ['prob'] for occ in occurrences]
			avg_prob = sum(probs) / len(probs)
			
			french_with_avg_prob.append({
				'word': fr_word,
				'avg_prob': avg_prob,
				'total_count': len(occurrences),
				'occurrences': occurrences
			})
		
		# sort by average probability, then by total count
		french_with_avg_prob.sort(key=lambda x: (x['avg_prob'], x['total_count']), reverse=True)
		
		french_trans_dict = {}
		for translation in french_with_avg_prob:
			french_trans_dict[translation['word']] = {
				'avg_prob': round(translation['avg_prob'], 4),
				'total_count': translation['total_count'],
				'occurrences': translation['occurrences']
			}
		
		final_structure[eng_word] = {
			'total_translation_count': len(french_translations),
			'french_translations': french_trans_dict
		}
	
	with open(word_pair_path_json, 'w', encoding='utf-8') as jsonfile:
		json.dump(final_structure, jsonfile, ensure_ascii=False, indent=2)
	
	total_english_words = len(english_groups)
	total_unique_pairs = sum(len(ft) for ft in english_groups.values())
	total_occurrences = len(all_word_pairs)
	
	return {
		'english_words': total_english_words,
		'unique_pairs': total_unique_pairs,
		'total_occurrences': total_occurrences
	}

