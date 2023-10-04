import random
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
from spacy.pipeline.spancat import DEFAULT_SPANCAT_MODEL
from tqdm import tqdm
import statistics

random.seed(10)

def offset_start(recipe_text, dot_indices):
	sentence_starts = [0]
	for i in range(len(dot_indices) - 1):
		if recipe_text[dot_indices[i] + 1] != " " :
			sentence_starts.append(dot_indices[i] + 1)
		elif recipe_text[dot_indices[i] + 2] != " ":
			sentence_starts.append(dot_indices[i] + 2)
		else:
			print("Further than 2")
	return sentence_starts

model_type = "ner"
training_set = []
span_key = "sc"
for j in range(100):
	txt_filename = f"recipe{j}.txt"
	ann_filename = f"recipe{j}.ann"
	with open(txt_filename) as txt_f, open(ann_filename) as ann_f:
		recipe_text = txt_f.read()
		annotations = ann_f.readlines()
	dot_indices = [i for i, ltr in enumerate(recipe_text) if ltr == "."]
	if model_type == "span":
		training_data = [(recipe_text[:dot_indices[0]], {"spans": {span_key: []}})] # changed "entities": to "spans":
	else:
		training_data = [(recipe_text[:dot_indices[0]], {"entities": []})]
	# training_data = [(recipe_text, {"spans": {span_key: []}})]
	# training_data = (recipe_text, {"spans": {span_key: []}})
	offsets = offset_start(recipe_text, dot_indices)
	for i in range(1, len(dot_indices)):
		if model_type == "span":
			training_data.append((recipe_text[dot_indices[i-1]+1:dot_indices[i]].strip(), {"spans": {span_key: []}}))
		else:
			training_data.append((recipe_text[dot_indices[i-1]+1:dot_indices[i]].strip(), {"entities": []}))
	annotations = [annotation.split() for annotation in annotations]
	for annotation in annotations:
		for i, dot_index in enumerate(dot_indices):
			if int(annotation[3]) <= dot_index:
				if model_type == "span":
					training_data[i][1]["spans"][span_key].append((int(annotation[2]) - offsets[i], int(annotation[3]) - offsets[i], annotation[1]))
				else:
					training_data[i][1]["entities"].append((int(annotation[2]) - offsets[i], int(annotation[3]) - offsets[i], annotation[1]))
				break
	training_set += training_data

if model_type == "span":
	nlp = spacy.blank("en")
	#spancat config - the definitions of each parameter are taken from spaCy's documentation 
	config = {
	    #this refers to the minimum probability to consider a prediction positive
	    "threshold": 0.5,
	    #the span key refers to the key in doc.spans 
	    "spans_key": span_key,
	    #this refers to the maximum number of labels to consider positive per span
	    "max_positive": None,
	     #a model instance that is given a list of documents with start end indices representing the labelled spans
	    "model": DEFAULT_SPANCAT_MODEL,
	    #A function that suggests spans. This suggester is fixed n-gram length of up to 3 tokens
	    "suggester": {"@misc": "spacy.ngram_suggester.v1", "sizes": [1, 2, 3]},
	}
	#add spancat component to nlp object
	nlp.add_pipe("spancat", config=config)
	#get spancat component 
	span = nlp.get_pipe('spancat')

	#Add labels to spancat component 
	labels = ["VERB", "WHAT", "WHERE", "HOW", "TIME", "TEMP"]
	for label in labels:
	    span.add_label(label)

	#get pipe you want to train on 
	pipe_exceptions = ["spancat"]
	unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

	# initialise spacy object 
	nlp.initialize()
	sgd = nlp.create_optimizer()

	#start training the spancat component 
	all_losses = []
	with nlp.disable_pipes(*unaffected_pipes):
	    for iteration in tqdm(range(500)):
	        # shuffling examples before every iteration
	        # random.shuffle(training_set)
	        losses = {}
	        batches = minibatch(training_set, size=compounding(4.0, 32.0, 1.001))
	        for batch in batches:
	            texts, anns = zip(*batch)
	            example = []
	            for i in range(len(texts)):
	            	doc = nlp.make_doc(texts[i])
	            	print(texts[i], anns[i])
	            	example.append(Example.from_dict(doc, anns[i]))
	            nlp.update(example, losses=losses, drop=0.1, sgd=sgd)
	        print("epoch: {} Losses: {}".format(iteration, str(losses)))
	        all_losses.append(losses['spancat'])
	nlp.to_disk("./trained_models/ner_500")
else:
	nlp = spacy.load("en_core_web_md")
	# nlp = spacy.blank("en")
	# nlp.add_pipe("ner")
	ner = nlp.get_pipe("ner")
	labels = ["VERB", "WHAT", "WHERE", "HOW", "TIME", "TEMP"]

	for label in labels:
		ner.add_label(label)

	pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
	unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

	num_iterations = 500
	# nlp.begin_training()
	with nlp.disable_pipes(*unaffected_pipes):
		for j in tqdm(range(num_iterations)):
			random.shuffle(training_set)
			losses = {}
			batches = minibatch(training_set, size=compounding(4.0, 32.0, 1.001))
			for batch in batches:
				batch_losses = []
				texts, anns = zip(*batch)
				example = []
				for i in range(len(texts)):
					doc = nlp.make_doc(texts[i])
					example.append(Example.from_dict(doc, anns[i]))
				nlp.update(
					example,
					drop=0.1,
					losses=losses)
				batch_losses.append(losses["ner"])
			print(f"Average batch loss: {statistics.mean(batch_losses)}")
	nlp.to_disk("./trained_models/ner_500_md")

	# test the model with an example instruction step
	doc = nlp("Place the baking pan in the oven and dice the potatoes.")
	{'VERB': ['Refrigerate'], 'WHAT': ['the lettuce'], 'WHERE': [], 'HOW': [], 'TIME': [], 'TEMP': []}