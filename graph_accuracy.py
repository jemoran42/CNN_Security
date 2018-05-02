import matplotlib.pyplot as plt

filename = "slurm-103364.out"
accuracies = []
loss = []
iterations = []

file = open(filename, "r")
for line in file:
	if 'Test net output #1: top1/acc = ' in line:
		start = line.find(' = ')
		end = line.find('\n')
		accuracies.append(float(line[start+3:end]))
	if 'Test net output #0: loss = ' in line:
		start = line.find(' = ')
		end = line.find(' (')
		loss.append(float(line[start+3:end]))
	if 'Testing net (#0)' in line:
		start = line.find('Iteration ')
		end = line.find(',')
		iterations.append(int(line[start+10:end]))
		
fig, ax1 = plt.subplots()
ax1.plot(iterations, accuracies, 'b-')
ax1.set_xlabel('Iterations')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_ylabel('Accuracy', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()
ax2.plot(iterations, loss, 'r-')
ax2.set_ylabel('Loss', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
plt.title('Accuracy and Loss')
plt.savefig('mike_103364.png',bbox_inches='tight')



