import matplotlib.pyplot as plt
import csv

filename = "accuracies.txt"

def convert_csv_to_arrays(filename, old_index, new_index):
    old_file = []
    new_file = []
 
    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
        	if 'ip1' in row[new_index]:
        		old_file.append(float(row[old_index]))
        	elif 'ip2' in row[new_index]:
        		new_file.append(float(row[old_index]))
 
    return old_file, new_file
	
accuracies_conv1,accuracies_conv2 = convert_csv_to_arrays(filename, 1, 2)
	
f = plt.figure(1)	
plt.plot(range(1,len(accuracies_conv1)+1),accuracies_conv1,'r')
plt.xlabel('Weights')
plt.ylabel('Accuracy')
plt.grid(True)
plt.axis([0,len(accuracies_conv1),0,1])

plt.title('Iris (ip1) - Accuracy vs Weights')
f.savefig('iris_ip1.png',bbox_inches='tight')

g = plt.figure(2)
plt.plot(range(1,len(accuracies_conv2)+1),accuracies_conv2,'r')
plt.xlabel('Weights')
plt.ylabel('Accuracy')
plt.grid(True)
plt.axis([0,len(accuracies_conv2),0,1])

plt.title('Iris (ip2) - Accuracy vs Weights')
g.savefig('iris_ip2.png',bbox_inches='tight')


