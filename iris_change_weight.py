import caffe
import os
import sys

sys.dont_write_bytecode = True

if len(sys.argv) < 1:
    print "Error: give input argument"
    sys.exit()
else:
    change_amount = float(sys.argv[1])/100.0

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
						# os.system("rm modified_weights.caffemodel")

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

	elif layer_type.type == 'InnerProduct':
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
				# os.system("rm modified_weights.caffemodel")

				log = open('log.txt',"r")
				save_file = open('accuracies_value_' + str(int(change_amount*100)) + '.txt',"a")

				for line in log:
					if 'caffe.cpp:330] accuracy = ' in line:
						start = line.find(' = ')
						end = line.find('\n')
						save_file.write(str(index_num) + ',' + str(line[start+3:end]) + ',layer_name=' + str(layer_name) + ',v=' + str(v) + ',u=' + str(u) + ',magnitude=' + str(net.params[layer_name][0].data[v][u]) + '\n')

				log.close()
				save_file.close()
				index_num += 1
