import os
import numpy as np
import pandas as pd
import math

def get_df(logfiles):
	""" Inserts the parameters of all the trained
		models (except BERT) into a dataframe. """
	fil_stats = {}

	for l in logfiles:
		dataset = l.split('_')[1]
		task = l.split('_')[-1]
		model = task.split("+")[0].split("=")[1]
		lambda_value = task.split("+")[1].split("=")[1]

		if 'anon' in l:
			anon = u'\u2718'
		else:
			anon = u'\u2714'

		try:
			seed = task.split("+")[2].split("=")[1]
		except:
			seed = 0

		fil_stats[l] = {'model':model,'Dataset':dataset,'LambdaV':lambda_value,'Seed':seed, 'Anon':anon}

	return pd.DataFrame().from_dict(fil_stats).T

def get_bert_df(bertfiles):
	""" Inserts the parameters of all the trained
		BERT models into a dataframe. """
	fil_stats = {}

	for l in bertfiles:
		dataset = l.split('_')[0]
		t = l.split('/')[0]
		if 'anon' in l:
			anon = u'\u2718'
			lamb = '0.0'
			m = t.replace('_anon', '')
		else:
			anon = u'\u2714'
			lamb = t.split('_')[3]
			m = t.replace('_'+lamb, '')

		model = m.replace(dataset+'_', '')
		model = model.replace('_'+model.split('_')[-1], '')
		try:
			if 'anon' in l:
				seed = '-'
				lamb = '0.0'
				seed = t.split('_')[-2].split('_')[-1]
			else:
				seed = t.split('_')[-1]

		except:
			seed = 0

		fil_stats[l] = {'model':model,'Dataset':dataset,'LambdaV':lamb, 'Seed':seed, 'Anon':anon}

	return pd.DataFrame().from_dict(fil_stats).T


def get_results(log_df, fil, directory):
	""" Extracts the accuracy and attention
		mass from a log fil. """
	log = pd.read_fwf(directory+fil,delimiter=' ')
	log = log[len(log)-7:]
	newlog_df = pd.Series([x for x in log.values])
	newlog_df = newlog_df.apply(lambda x:x[0])
	counter = 0
	acc = 0
	am = 0

	for y in newlog_df.values:
		for stat in ['best test accuracy' ,'attention_ratio']:

			if type(y) == str:
				if stat in y:

					y = y.replace(",","")
					iteration = int(y.split('iter', maxsplit=1)[-1].split(maxsplit=1)[0][:-1])
					stat1_val = y.split(stat, maxsplit=1)[-1].split(maxsplit=1)[-1].split(maxsplit=1)[0].replace("=","")
					stat1_val = float(stat1_val)*100

					if stat == 'best test accuracy':
						acc = stat1_val
					else:
						am = stat1_val
	return acc, am

def get_bert_results(log_df, fil, directory):
	""" Extracts the accuracy and attention
		mass from a log fil for the BERT model. """

	lines = open(directory+fil, "r").readlines()
	acc = float(lines[0].split("=")[1][0:-1])*100
	if 'bert_max' in fil:
		am = float(lines[2].split("=")[1][0:-1])*100
	else:
		am = float(lines[4].split("=")[1][0:-1])*100
	return acc, am

def round_correctly(outputs):
	""" Rounds the values in order to optimize readability of the table. """
	for i in range(len(outputs)):
		if outputs[i] == 0.0 or round(outputs[i],2) != 0.0:
			outputs[i] = round(outputs[i],2)
		else:
			if i == 0 or i == 2:
				outputs[i] = "{:.2e}".format(outputs[i])
			else:
				outputs[i] = ""
	return outputs

def make_df(log_df, bert_df, log_dir, bert_dir, simple=False):
	""" Inserts the mean accuracy and std over all
		seeds of each trained model into a dataframe. """
	final_df = []

	# Find Acc. and A.M. for all seeds of models with the same parameters
	for df in [log_df, bert_df]:
		# print(df)
		for model in df.model.unique():

			# print(model, df.model.unique())
			for anon in [u'\u2718', u'\u2714']:
				for lamb in  ['1.0', '0.0', '0.1']:
					stats = [model, lamb, anon]
					if 'bert' in model:
						datasets = ['pronoun', 'occupation-classification', 'sst-wiki']
					else:
						datasets = df.Dataset.unique()

					for ds in datasets:
						accs = []
						ams = []

						# Sub-df only containing the seeds of one model.
						temp = df.loc[(df['model'] == model) & (df['Dataset'] == ds) & (df['LambdaV'] == lamb) & (df['Anon'] == anon)]

						if len(temp) > 0:
							for fil in temp.index:
								if 'bert' in model:
									acc, am = get_bert_results(temp, fil, bert_dir)
								else:
									acc, am = get_results(temp, fil, log_dir)

								# Some files did not output correctly and contain nan
								if not math.isnan(acc) and not math.isnan(am):
									accs.append(acc)
									ams.append(am)

							# Means and std's accross seeds
							outputs = [np.mean(accs), np.std(accs), np.mean(ams), np.std(ams)]

							# Change to scientific notation for values that are rounded to 0.0
							outputs = round_correctly(outputs)
							# print(fil, outputs)

							if simple:
								stats.append(str(outputs[0]))
								if anon == u'\u2718':
									stats.append('-')
								else:
									stats.append(str(outputs[2]))

							else:
								stats.append(str(outputs[0])+ u" \u00B1 "+ str(outputs[1]))

								# Models where impermissible tokens are removed have no Atttention Mass
								if anon == u'\u2718':
									stats.append('-')
								else:
									stats.append(str(outputs[2])+u" \u00B1 "+ str(outputs[3]))
							# print(stats)
					if len(temp) > 0:
						final_df.append(stats)


	total_df = pd.DataFrame(final_df, columns = ['Model','\u03BB', 'I', 'Gender-Acc.', 'Gender-A.M.', 'Occupation-Acc.', 'Occupation-A.M.', 'SST-Acc.', 'SST-A.M.'])
	total_df = total_df[['Model', '\u03BB', 'I','Occupation-Acc.', 'Occupation-A.M.', 'Gender-Acc.', 'Gender-A.M.','SST-Acc.', 'SST-A.M.']]
	total_df = total_df.replace({'emb-lstm-att': 'BiLSTM', 'emb-att': 'Embedding', 'bert_max': 'BERT (max)', 'bert_mean': 'BERT (mean)'})
	total_df = total_df.sort_values(by = ['Model','\u03BB', 'I'], ascending = [False, True, False])

	return total_df


def print_table(logfiles, bertfiles, log_dir, bert_dir, simple):
	""" Returns a table that presents the mean accuracy
		and std over all seeds of each trained model. """
	log_df = get_df(logfiles)

	bert_df = get_bert_df(bertfiles)

	return make_df(log_df, bert_df, log_dir, bert_dir, simple)

def calc_runtime(dfs, log_dir, bert_dir):
	""" Displays a table containing the average training time in minutes
		for one seed for every model (for every paramet and dataset). """
	total_df = pd.DataFrame()
	avg_time = []
	for df in dfs:
		for model in df.model.unique():
			for anon in df.Anon.unique():

				for lamb in df.LambdaV.unique():
					stats = [model, lamb, anon]

					for ds in df.Dataset.unique():
						# Sub-df only containing the seeds of one model.
						temp = df.loc[(df['model'] == model) & (df['Dataset'] == ds) & (df['LambdaV'] == lamb) & (df['Anon'] == anon)]
						times = []
						if len(temp) > 0:
							for fil in temp.index:
								if 'bert' in model:
									lines = open(bert_dir+fil, "r").readlines()
								else:
									lines = open(log_dir+fil, "r").readlines()

								time = [float(x.split('=')[-1].split('s\n')[0]) for x in lines if 'time' in x]
								times.append(np.round(np.sum(time)/60,1))

							stats.append(np.mean(times))
					if len(temp) > 0:
						avg_time.append(stats)

	total_df = pd.DataFrame(avg_time, columns = ['Model','\u03BB', 'I', 'Gender Classification', 'Occupation Classification', 'Sentiment Analysis'])
	total_df = total_df[['Model','\u03BB', 'I', 'Occupation Classification', 'Gender Classification', 'Sentiment Analysis']]
	total_df = total_df.replace({'emb-lstm-att': 'BiLSTM', 'emb-att': 'Embedding', 'bert_max': 'BERT (max)', 'bert_mean': 'BERT (mean)'})
	total_df = total_df.sort_values(by = ['Model','\u03BB', 'I'], ascending = [False, True, False])

	return total_df


def create_difference_table(simple):
	""" Creates a table that displays the differences
		between our results and the one from Pruthi et al. """

	# Table 3 from the paper by Pruthi et al
	orig = [['Embedding', 0.0, u'\u2718', 93.8, '-' , 66.8 , '-' , 48.9, '-'],
	['Embedding', 0.0, u'\u2714', 93.3, 51.4 , 100, 99.2, 70.7, 48.4],
	['Embedding', 0.1, u'\u2714', 96.2, 4.6, 99.4, 3.4, 67.9, 36.4],
	['Embedding', 1.0, u'\u2714', 96.2, 1.3, 99.2, 0.8, 48.4, 8.7],
	['BiLSTM', 0.0, u'\u2718', 93.3, '-' , 63.3 , '-' , 49.1, '-'],
	['BiLSTM', 0.0, u'\u2714', 96.4, 50.3 , 100, 96.8, 76.9, 77.7],
	['BiLSTM', 0.1, u'\u2714', 96.4, 0.08, 100, 10e-6, 60.6, 0.04],
	['BiLSTM', 1.0, u'\u2714', 96.7, 10e-3, 100, 10e-6, 61.0, 0.07],
	['BERT (mean)', 0.0, u'\u2718', 95.0, '-' , 72.8 , '-' , 50.4, '-'],
	['BERT (mean)', 0.0, u'\u2714', 97.2, 13.9 , 100, 80.8, 90.8, 59.0],
	['BERT (mean)', 0.1, u'\u2714', 97.2, 0.01, 99.9, 10e-3, 90.9, 10e-2],
	['BERT (mean)', 1.0, u'\u2714', 97.2, 10e-3, 99.9, 10e-3, 90.6, 10e-3],
	['BERT (max)', 0.0, u'\u2718', 95.0, '-' , 72.8 , '-' , 50.4, '-'],
	['BERT (max)', 0.0, u'\u2714', 97.2, 99.7, 100, 99.7, 90.8, 96.2],
	['BERT (max)', 0.1, u'\u2714', 97.1, 10e-3, 99.9, 10e-3, 90.7, 10e-2],
	['BERT (max)', 1.0, u'\u2714', 97.4, 10e-3, 99.8, 10e-4, 90.2, 10e-3],
	]

	orig_df = pd.DataFrame(orig, columns=['Model', '\u03BB', 'I','Occupation-Acc.', 'Occupation-A.M.', 'Gender-Acc.', 'Gender-A.M.','SST-Acc.', 'SST-A.M.'])

	differences = []
	for i in range(simple.values.shape[0]):

		# Model parameters
		diff = list(simple.values[i][:3])

		# Accuracies and A.M.s
		x = orig_df.values[i][3:]
		y = simple.values[i][3:]

		# Calculate difference between values
		vals = [round(float(x[a])-float(y[a]),2) if x[a] != '-' and y[a] != '-' else '-' for a in range(len(x))]

		diff.extend(vals)

		differences.append(diff)

	pd.set_option("display.max_rows", None, "display.max_columns", None)
	diff_df = pd.DataFrame(differences, columns = ['Model', '\u03BB', 'I','Occupation-Acc.', 'Occupation-A.M.', 'Gender-Acc.', 'Gender-A.M.','SST-Acc.', 'SST-A.M.'])

	return diff_df

