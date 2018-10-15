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
					#inputs,label
					# TODO(H): Check if indexs match with csv files
					features, labels = row[:-1], row[-1] 
					ex = tf.train.Example()
					# TODO(H): Ask for correct datatypes and change them
					ex.features.feature["features"].float_list.value.extend(features)
					ex.features.feature["labels"].float_list.value.append(labels) 
					writer.write(ex.SerializeToString())		
