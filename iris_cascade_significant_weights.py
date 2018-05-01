#########################################################################
# Brute force method with 100% reduction of weights                     #
# Sort weight index by accuracy drop                                    #
# Cascade Significant weights to 0 until specified accuracy is acheived #
#########################################################################

# THIS SCRIPT ONLY WORKS ON FULLY CONNECTED LAYERS CURRENTLY

import caffe
import os
import sys
import csv


# pulls relevant data from CSV files; returns accuracies and weight magnitude for two layers in network
def convert_csv_to_arrays(filename, first_index, second_index, third_index, fourth_index, layer_name):
    index = []
    accuracy = []
    v = []
    u = []

    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if last_layer_name in row[layer_name]:     # change string to name of one layer in accuracy files
                index.append(float(row[first_index]))
                accuracy.append(float(row[second_index]))
                v.append(abs(int(row[third_index][(row[third_index].find('=')+1):])))
                u.append(abs(int(row[fourth_index][(row[fourth_index].find('=')+1):]))) 

    return index, accuracy, v, u


##########################################################
# BRUTE FORCE TO FIND SIGNIFICANT WEIGHTS

sys.dont_write_bytecode = True


# usage for iris network:		python iris_cascade_significant_weights.py 0 75 ip2
# modify weight to 0, cascade until accuracy of 75% is acheived, layer modified is ip2

if len(sys.argv) < 3:
    print "Error: give input arguments"
    sys.exit()
else:
    change_amount = float(sys.argv[1])/100.0
    desired_accuracy = float(sys.argv[2])/100.0
    last_layer_name = str(sys.argv[3])

caffe.set_mode_gpu()
caffe.set_device(0)

path = '/local/tmp/jem_iris_value/'
prototxt = path + 'iris_caffe_network.prototxt'
weights = path + '_iter_1000.caffemodel'
modified_weights = path + 'modified_weights.caffemodel'

net = caffe.Net(prototxt, weights, caffe.TEST)
new_net = caffe.Net(prototxt, caffe.TEST)

index_num = 0
current_layer = ' '

for layer_name, layer_type in zip(net._layer_names, net.layers):
	if layer_type.type == 'Convolution':
	# if layer_name == 'conv1':
		a,b,c,d = net.params[layer_name][0].data.shape
		for z in range(0,a):
			for y in range(0,b):
				for x in range(0,c):
					for w in range(0,d):
						for reset_name, reset_type in zip(net._layer_names, net.layers):
							if reset_type.type == 'Convolution' or reset_type.type == 'InnerProduct':
								new_net.params[reset_name][0].data[...] = net.params[reset_name][0].data[...]
								new_net.params[reset_name][1].data[...] = net.params[reset_name][1].data[...]
						print layer_name
						print 'z = ' + str(z) + ' y = ' + str(y) + ' w = ' + str(w) + ' x = ' + str(x)
						print new_net.params[layer_name][0].data[z][y][x][w]
						new_net.params[layer_name][0].data[z][y][x][w] = new_net.params[layer_name][0].data[z][y][x][w]*change_amount
						print new_net.params[layer_name][0].data[z][y][x][w]
						new_net.save(modified_weights)

						files = os.listdir(path)
						for file in files:
							if 'modified_weights.caffemodel' in file:
								print 'Weights found'

						os.system("ls")
						os.system("caffe test --model=" + prototxt + " --weights=modified_weights.caffemodel --gpu=0 2>&1 | tee log.txt")
						os.system("rm modified_weights.caffemodel")

						log = open('log.txt',"r")
						save_file = open('accuracies_' + str(int(change_amount*100)) + '.txt',"a")

						for line in log:
							if 'caffe.cpp:330] accuracy = ' in line:
								start = line.find(' = ')
								end = line.find('\n')
								save_file.write(str(index_num) + ',' + str(line[start+3:end]) + ',layer_name=' + str(layer_name) + ',z=' + str(z) + ',y=' + str(y) + ',x=' + str(x) + ',w=' + str(w) + '\n')

						log.close()
						save_file.close()
						index_num += 1

	# elif layer_type.type == 'InnerProduct':
	elif layer_name == last_layer_name:
		e,f = net.params[layer_name][0].data.shape
		for v in range(0,e):
			for u in range(0,f):
				for reset_name, reset_type in zip(net._layer_names, net.layers):
					if reset_type.type == 'Convolution' or reset_type.type == 'InnerProduct':
						new_net.params[reset_name][0].data[...] = net.params[reset_name][0].data[...]
						new_net.params[reset_name][1].data[...] = net.params[reset_name][1].data[...]
				print layer_name
				new_net.params[layer_name][0].data[v][u] = new_net.params[layer_name][0].data[v][u]*change_amount
				new_net.save(modified_weights)

				files = os.listdir(path)
				for file in files:
					if 'modified_weights.caffemodel' in file:
						print 'Weights found'

				os.system("ls")
				os.system("caffe test --model=" + prototxt + " --weights=modified_weights.caffemodel --gpu=0 2>&1 | tee log.txt")
				os.system("rm modified_weights.caffemodel")

				log = open('log.txt',"r")
				save_file = open('accuracies_value_' + str(int(change_amount*100)) + '.txt',"a")

				for line in log:
					if 'caffe.cpp:330] accuracy = ' in line:
						start = line.find(' = ')
						end = line.find('\n')
						save_file.write(str(index_num) + ',' + str(line[start+3:end]) + ',layer_name=' + str(layer_name) + ',v=' + str(v) + ',u=' + str(u) + ',magnitude=' + str("{0:.10f}".format(net.params[layer_name][0].data[v][u])) + '\n')

				log.close()
				save_file.close()
				index_num += 1

################################################################
# SORT THE WEIGHT INDEXES BY ACCURACY REDUCTION (SIGNIFICANCE)

filename = "accuracies_0.txt"

index,accuracy,v,u = convert_csv_to_arrays(filename, 0, 1, 3, 4, 2)

zipped_list = zip(index,accuracy,v,u)

sorted_zipped_list = sorted(zipped_list, key=lambda x: x[1])

sorted_index,sorted_accuracy,sorted_v,sorted_u = zip(*sorted_zipped_list)


###############################################################
# CASCADE SIGNIFICANT WEIGHTS UNTIL DESIRED ACCURACY REACHED (desired_accuracy)

# RESET WEIGHTS TO ORIGINAL
for reset_name, reset_type in zip(net._layer_names, net.layers):
	if reset_type.type == 'Convolution' or reset_type.type == 'InnerProduct':
		new_net.params[reset_name][0].data[...] = net.params[reset_name][0].data[...]
		new_net.params[reset_name][1].data[...] = net.params[reset_name][1].data[...]


desired = False

# CASCADE CHANGES
for x,y in zip(sorted_v,sorted_u):
	new_net.params[last_layer_name][0].data[x][y] = new_net.params[last_layer_name][0].data[x][y]*change_amount
	new_net.save(modified_weights)

	os.system("caffe test --model=" + prototxt + " --weights=modified_weights.caffemodel --gpu=0 2>&1 | tee log.txt")

	log = open('log.txt', "r")

	for line in log:
		if 'caffe.cpp:330] accuracy = ' in line:
			start = line.find(' = ')
			end = line.find('\n')
			current_accuracy = float(line[start+3:end])

	if current_accuracy <= desired_accuracy:
		desired = True
		break

if desired:
	print 'Desired accuracy found. File saved as ' + modified_weights

else:
	print 'Desired accuracy could not be acheived. Try smaller accuracy drop.'

###############################################################


# number of iterations, weight index x, weight index y, accuracy for individual weight, accuracy from accumulation

# GoogleNet 
# 8 class based datasets (hdf5)
# 8 class based plots (brute force)
# 1 class based cascade to prove killing one class in network

# Iris and LeNet as well