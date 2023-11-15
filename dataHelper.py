from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset, concatenate_datasets
import json
import os





def get_dataset(dataset_name, sep_token):
	'''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''

	# aggregation dataset
	if type(dataset_name) == list:
		dataset = get_dataset(dataset_name[0], sep_token)
		for name in dataset_name:
			if name == dataset_name[0]:
				continue
			tmp_dataset = get_dataset(name, sep_token)


			# add label number to tmp_dataset
			cur_label_num = max(dataset["train"]["label"])

			tmpdict = Dataset.to_dict(tmp_dataset["train"])
			for i in range(0,len(tmpdict["label"])):
				tmpdict["label"][i] += cur_label_num
			tmp_dataset["train"] = Dataset.from_dict(tmpdict)

			tmpdict = Dataset.to_dict(tmp_dataset["test"])
			for i in range(0,len(tmpdict["label"])):
				tmpdict["label"][i] += cur_label_num
			tmp_dataset["test"] = Dataset.from_dict(tmpdict)





			dataset_train = concatenate_datasets([tmp_dataset['train'], dataset['train']])
			dataset_test = concatenate_datasets([tmp_dataset['test'], dataset['test']])
			dataset["train"] = dataset_train
			dataset["test"] = dataset_test

		
		return dataset


	# SemEval14 dataset
	if dataset_name == "restaurant_sup" or dataset_name == "laptop_sup" or dataset_name == "restaurant_fs" or dataset_name == "laptop_fs":

		if dataset_name == "restaurant_sup"or dataset_name == "restaurant_fs":
			dataset_filename = "SemEval14-res"
		elif dataset_name == "laptop_sup" or dataset_name == "laptop_fs":
			dataset_filename = "SemEval14-laptop"
		

		# prepare path
		train_path =  os.getcwd() + '\\'  + dataset_filename + '\\' + 'train.json'
		test_path = os.getcwd() + '\\'  + dataset_filename + '\\' + 'test.json'


		# generate train dataset
		f = open(train_path,'r')
		fj = json.load(f)  #fj is already a dict

		# build up polarity to idx dict
		idx={}
		cnt=0
		for key in fj.keys():
			if fj[key]["polarity"] not in idx.keys():
				idx[fj[key]["polarity"]] = cnt
				cnt += 1


		
		text = []
		label = []
		for key in fj.keys():
			text.append(fj[key]["sentence"]+sep_token+fj[key]["term"])
			label.append(idx[fj[key]["polarity"]])
		train_dict = {}
		train_dict["text"] = text
		train_dict["label"] = label
		
		train_dataset = Dataset.from_dict(train_dict)

		

		# generate test dataset
		f = open(test_path,'r')
		fj = json.load(f)  #fj is already a dict


		
		text = []
		label = []
		for key in fj.keys():
			text.append(fj[key]["sentence"]+sep_token+fj[key]["term"])
			label.append(idx[fj[key]["polarity"]])
		test_dict = {}
		test_dict["text"] = text
		test_dict["label"] = label
		
		test_dataset = Dataset.from_dict(test_dict)


		# merge two into a dictset
		dataset_dict = {}
		dataset_dict["train"] = train_dataset
		dataset_dict["test"] = test_dataset
		dataset=DatasetDict(dataset_dict)


		# print(dataset["train"][0])





	# ACL dataset
	elif dataset_name == "acl_sup" or dataset_name == "acl_fs":

		dataset = load_dataset("json", data_files={"train": "ACL-ARC\\"+"train.jsonl", "test": "ACL-ARC\\" + "test.jsonl"})

		train_dict = Dataset.to_dict(dataset["train"])
		test_dict = Dataset.to_dict(dataset["test"])

		train = {}
		train['text'] = train_dict['text']
		train['label'] = train_dict['label']
		test = {}
		test['text'] = test_dict['text']
		test['label'] = test_dict['label']

		idx = {}
		cnt = 0
		for label in train['label']:
			if label not in idx.keys():
				idx[label] = cnt
				cnt += 1

		for i in range(0,len(train['label'])):
			train['label'][i] = idx[train['label'][i]]


		for i in range(0,len(test['label'])):
			test['label'][i] = idx[test['label'][i]]


		train = Dataset.from_dict(train)
		test = Dataset.from_dict(test)
		dataset_dict = {}
		dataset_dict["train"] = train
		dataset_dict["test"] = test
		dataset=DatasetDict(dataset_dict)

		
		

		

		
		
		

		
	# prepare agnews dataset
	elif dataset_name == "agnews_sup" or dataset_name == "agnews_fs":


		dataset = Dataset.from_file("AGnews\\data.arrow")
		dataset = dataset.train_test_split(test_size=0.1,seed=2022)
		

	else:
		print("This dataset is not supported")
		return



	# deal with few-shot dataset
	# select 32 samples from the original dataset and ensure the number of labels is balanced.
	if dataset_name[-2:]=="fs":	
		train_fs = dataset["train"].shuffle().select(range(32))
		test_fs = dataset["test"].shuffle().select(range(32))
		dataset_dict = {}
		dataset_dict["train"] = train_fs
		dataset_dict["test"] = test_fs
		dataset=DatasetDict(dataset_dict)
		


	return dataset


