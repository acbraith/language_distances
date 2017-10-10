import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE,MDS
from itertools import combinations

from memoize import persistent_memoize

#@persistent_memoize('load_lang')
def load_lang(language):
	data = pd.read_csv('dataset.tab', sep='\t')
	word_list = list(data)[-100:]
	row = data[data.names == language]
	lang_words = []
	for word in word_list:
		vals = [] if row[word].isnull().any() else row[word].max().split()
		lang_words += [vals]
	return lang_words

def LD(s1,s2):
	# https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
	if len(s1) < len(s2):
		return LD(s2, s1)

	# len(s1) >= len(s2)
	if len(s2) == 0:
		return len(s1)

	previous_row = range(len(s2) + 1)
	for i, c1 in enumerate(s1):
		current_row = [i + 1]
		for j, c2 in enumerate(s2):
			insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
			deletions = current_row[j] + 1       # than s2
			substitutions = previous_row[j] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row

	return previous_row[-1]

def LDN(a,b):
	# Evaluating linguistic distance measures, Wichmann et al 2010
	# todo implement LDND
	return LD(a,b) / max(len(a),len(b))

def word_distance(ws):
	w1s,w2s = ws
	if len(w1s) == 0 or len(w2s) == 0:
		return np.nan
	ds = []
	for i in range(len(w1s)):
		for j in range(len(w2s)):
			ds += [LDN(w1s[i],w2s[j])]
	return np.mean(ds)

@persistent_memoize('lang_distance')
def lang_distance(l1, l2):
	dists = map(word_distance, zip(load_lang(l1),load_lang(l2)))
	return np.mean([d for d in dists if np.isfinite(d)])

@persistent_memoize('languages_distance_matrix')
def languages_distance_matrix(languages):
	dm = np.zeros((len(languages),len(languages)))
	for i in range(0,len(languages)):
		for j in range(i+1,len(languages)):
			d = lang_distance(languages[i],languages[j])
			dm[i,j] = d
			dm[j,i] = d
	return dm

def get_manifold(dm, dimensions=2, mds=True):
	# MDS gives much better looking results than TSNE
	if mds:
		X = MDS(
			n_components=dimensions, 
			metric=True, dissimilarity='precomputed',n_init=600,n_jobs=6).fit_transform(dm)
	else:
		X = TSNE(
			n_components=dimensions, 
			perplexity=5, metric='precomputed', n_iter=10000).fit_transform(dm)
	return X

def visualise_languages(languages,plot_clustering=1,line_fading=2):
	# plot_clustering: for higher values, plot will be more clustered
	# line_fading: for higher values, lines between more distant languages will fade more
	print("Building DM")
	dm = languages_distance_matrix(languages)
	print("Building Manifold")
	X = get_manifold(dm**plot_clustering)
	print("Plotting")
	plt.scatter(*zip(*X),s=0)
	# draw lines between pairs
	for pair in combinations(languages,2):
		i = languages.index(pair[0])
		j = languages.index(pair[1])
		plt.plot((X[i][0],X[j][0]),(X[i][1],X[j][1]),
			'k-',lw=3,alpha=(1-dm[i,j])**line_fading)
	# label points
	bbox_props = dict(boxstyle="round,pad=0.3", fc="pink", ec="k", lw=2)
	for i,lang in enumerate(languages):
		plt.text(X[i][0],X[i][1],
			lang.replace('_','\n').title(), 
			ha='center',va='center',
			bbox=bbox_props)
	plt.show()

duolingo = [
	'ENGLISH',
	'SPANISH','FRENCH','STANDARD_GERMAN',
	'ITALIAN','PORTUGUESE','RUSSIAN',
	'DUTCH','SWEDISH','IRISH_GAELIC',
	'TURKISH','NORWEGIAN_BOKMAAL','DANISH',
	'POLISH','HEBREW','VIETNAMESE',
	'GREEK','ESPERANTO','UKRAINIAN',
	'WELSH','HUNGARIAN','SWAHILI',
	'ROMANIAN','CZECH',# No High Valyrian
	'KLINGON','KOREAN','INDONESIAN',
	'JAPANESE','MANDARIN','HINDI',
	]

visualise_languages(duolingo)