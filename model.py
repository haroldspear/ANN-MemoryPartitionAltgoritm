import tensorflow as tf
import pandas as pd
import os

DNNClassifier=""
#This function transform csv files to tfrecords, to improve performance of algoritms.
#To generate again and replace files send 1 as parameter
def csvToTfRecords(replace=0, csv_dir="traces_csv", tfr_dir="traces_tfrecords"):
	traces_csv=os.listdir(csv_dir)
	traces_tfrecords=os.listdir(tfr_dir)
	generate=False
	for trace_csv in traces_csv:
		if(replace==1):
			generate=True
		elif(replace==0):
			if(trace_csv.split(".")[0]+".tfrecords" not in traces_tfrecords):
				generate=True
		if(generate):
			csv_file=pd.read_csv(csv_dir+"/"+trace_csv).values
			with tf.python_io.TFRecordWriter(tfr_dir+"/"+trace_csv.split(".")[0]+".tfrecords") as writer:
				for row in csv_file:
					#inputs,labels
					# TODO(H): Check if indexs match with csv files
					inputs, labels = row[:-4], row[-4] 
					ex = tf.train.Example()
					# TODO(H): Ask for correct datatypes and change them
					ex.features.feature["inputs"].float_list.value.extend(inputs)
					ex.features.feature["labels"].float_list.value.extend(labels) 
					writer.write(ex.SerializeToString())

def parse_fn(record):
	#Transform string data to features
	# TODO(H): Change datatypes if is necessary when test with real dataset, and using tf.cast or tf.convert_to_tensor
	features={
		'inputs': tf.FixedLenFeature([], tf.float),
		'labels': tf.FixedLenFeature([], tf.float),
	}

	parsed = tf.parse_single_example(record, features)

	inputs = tf.cast(parsed['inputs'], float64)
	labels = tf.cast(parsed['labels'], float64)

	return {'inputs': inputs}, {'labels': labels}



def input_fn(tfr_dir="traces_tfrecords"):
	#get dataset
	dataset = (
		tf.data.TFRecordDataset(os.listdir(tfr_dir))
		.map(parse_fn)
		.batch(1024)
		)

	iterator = dataset.make_one_shot_iterator()

	inputs, labels = iterator.get_next()

	#Return a dictionary of inputs and labels
	return inputs, labels

def initialize_estimator():
	#Estimator
	global DNNClassifier
	DNNClassifier = tf.estimator.DNNClassifier(
		feature_columns = [tf.feature_column.numeric_column(key='inputs', dtype=tf.float64, shape=(377,))],
		hidden_units = [256, 256, 256, 256],
		n_classes = 96,
		model_dir = '/tmp/tf_model'
		)

def train_and_evaluate(tfr_train_dir="traces_tfrecords/train", tfr_eval_dir="traces_tfrecords/train" )
	#Spec
	global DNNClassifier
	train_s = tf.estimator.TrainSpec(input_fn = lambda: input_fn(tfr_train_dir) , max_steps=1000)
	eval_s = tf.estimator.EvalSpec(input_fn = lambda: input_fn(tfr_eval_dir) )

	tf.estimator.train_and_evaluate(DNNClassifier, train_s, eval_s)

#TODO(H): Create main
#Predictions:
train_and_evaluate()
predictions = list(DNNClassifier.predict(input_fn = lambda: input_fn('traces_tfrecords/test')))
