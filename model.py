import tensorflow as tf
import pandas as pd
import os

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

