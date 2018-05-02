###############################################
# THIS MUST BE RAN WITH PYTHON3 ON A LINUX PC #
###############################################


import matplotlib.pyplot as plt
import csv

dataset = ""    # used if accuracies are dataset specific (EX. just class c0 for iris network); otherwise leave as empty string
directory = "./accuracies/"  # directory where accuracy files are stored

# filenames for each reduction percentage
filename0 = directory + "accuracies" + dataset + "_0.txt"  
filename25 = directory + "accuracies" + dataset + "_25.txt"
filename50 = directory + "accuracies" + dataset + "_50.txt"
filename75 = directory + "accuracies" + dataset + "_75.txt"
filename100 = directory + "accuracies" + dataset + "_100.txt"

# pulls relevant data from CSV files; returns accuracies and weight magnitude for two layers in network
def convert_csv_to_arrays(filename, old_index, new_index, mag_index):
    old_file = []
    new_file = []
    magnitude_ip1 = []
    magnitude_ip2 = []

    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if 'loss3/classifier_type' in row[new_index]:     # change string to name of one layer in accuracy files
                old_file.append(float(row[old_index]))
                magnitude_ip1.append(abs(float(row[mag_index][(row[mag_index].find('=')+1):(row[mag_index].find('\n'))])))
            elif 'ip2' in row[new_index]:   # change string to name of another layer in accuracy files
                new_file.append(float(row[old_index]))
                magnitude_ip2.append(abs(float(row[mag_index][(row[mag_index].find('=')+1):(row[mag_index].find('\n'))]))) 

    return old_file, new_file, magnitude_ip1, magnitude_ip2
	
accuracies0_ip1,accuracies0_ip2,magnitude_ip1,magnitude_ip2 = convert_csv_to_arrays(filename0, 1, 2, 5)
accuracies25_ip1,accuracies25_ip2,magnitude_ip1,magnitude_ip2 = convert_csv_to_arrays(filename25, 1, 2, 5)
accuracies50_ip1,accuracies50_ip2,magnitude_ip1,magnitude_ip2 = convert_csv_to_arrays(filename50, 1, 2, 5)
accuracies75_ip1,accuracies75_ip2,magnitude_ip1,magnitude_ip2 = convert_csv_to_arrays(filename75, 1, 2, 5)
accuracies100_ip1,accuracies100_ip2,magnitude_ip1,magnitude_ip2 = convert_csv_to_arrays(filename100, 1, 2, 5)

# Generates a single figure with multiple y-axis
f, ax1 = plt.subplots()	
ax1.plot(range(1,len(accuracies0_ip1)+1),accuracies0_ip1,ms=5,label="0%")
ax1.plot(range(1,len(accuracies25_ip1)+1),accuracies25_ip1,ms=5,label="25%")
ax1.plot(range(1,len(accuracies50_ip1)+1),accuracies50_ip1,ms=5,label="50%")
ax1.plot(range(1,len(accuracies75_ip1)+1),accuracies75_ip1,ms=5,label="75%")
ax1.plot(range(1,len(accuracies100_ip1)+1),accuracies100_ip1,ms=5,label="100%")
ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax1.set_xlabel('Weights')
ax1.set_ylabel('Accuracy')
ax1.grid(True)
ax1.axis([0,len(accuracies0_ip1),0.75,0.78])

ax2 = ax1.twinx()
ax2.plot(range(1,len(accuracies0_ip1)+1),magnitude_ip1,'m--',label="Weight Magnitude")
ax2.axis([0,len(accuracies0_ip1),0.0,1.1])
ax2.legend(bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.)

f.tight_layout()

# Title on Graph
plt.title('GoogleNet (last layer): ' + dataset)
# Save figure with filename 
f.savefig('googlenet_last_layer_value' + dataset +'.png',bbox_inches='tight')


# # Generates a second figure with same idea as above
# g, ax1 = plt.subplots()
# ax1.plot(range(1,len(accuracies0_ip2)+1),accuracies0_ip2,ms=5,label="0%")
# ax1.plot(range(1,len(accuracies25_ip2)+1),accuracies25_ip2,ms=5,label="25%")
# ax1.plot(range(1,len(accuracies50_ip2)+1),accuracies50_ip2,ms=5,label="50%")
# ax1.plot(range(1,len(accuracies75_ip2)+1),accuracies75_ip2,ms=5,label="75%")
# ax1.plot(range(1,len(accuracies100_ip2)+1),accuracies100_ip2,ms=5,label="100%")
# ax1.set_xlabel('Weights')
# ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# ax1.set_ylabel('Accuracy')
# ax1.grid(True)
# ax1.axis([0,len(accuracies0_ip2),0.0,1.1])

# ax2 = ax1.twinx()
# ax2.plot(range(1,len(accuracies0_ip2)+1),magnitude_ip2,'m--',label="Weight Magnitude")
# ax2.axis([0,len(accuracies0_ip2),0.0,1.1])
# ax2.legend(bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.)

# g.tight_layout()

# # Title on Graph
# plt.title('LeNet (ip2): ' + dataset)
# # Save figure with filename
# g.savefig('lenet_ip2_value' + dataset + '.png',bbox_inches='tight')


