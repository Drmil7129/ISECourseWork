import json
from scipy.stats import wilcoxon
import numpy as np

linear = json.load(open('linear_regression_results.txt'))
neural = json.load(open('neural_network_results.txt'))
linear_count = 0
neural_count = 0
system_list = []
linear_errors = {}
neural_errors = {}
linear_cents = {}
neural_cents = {}
neural_cents_array =[]
linear_cents_array =[]
systems = ['batlik', 'dconvert', 'h2', 'jump3r', 'kanzi', 'lrzip', 'x264', 'xz', 'z3']
systems_length = {'batlik':11, 'dconvert':12, 'h2':8, 'jump3r':6, 'kanzi':9, 'lrzip':13, 'x264':9, 'xz':13, 'z3':12}
file = open('comparison_results.txt','w')
for entry in systems:
    linear_errors[entry] = 0.0
    neural_errors[entry] = 0.0
    linear_cents[entry] = 0.0
    neural_cents[entry] = 0.0

for entry in linear:
    current_system = "bla"
    for system in systems:
        if system in entry:
            current_system = system
    linear_mae = linear[entry][0]
    neural_mae = neural[entry][0]
    linear_mape = linear[entry][1]
    neural_mape = neural[entry][1]
    linear_errors[current_system] += linear_mae
    neural_errors[current_system] += neural_mae
    linear_cents[current_system] += linear_mape
    linear_cents_array.append(linear_mape)
    neural_cents[current_system] += neural_mape
    neural_cents_array.append(neural_mape)
    print(entry)
    print("Linear MAE : {:.2f}  Neural MAE :{:.2f}".format(linear_mae,neural_mae))
    print("Linear MAPE : {:.2f}  Neural MAPE :{:.2f}\n".format(linear_mape, neural_mape))
    file.write(entry)
    file.write("\n")
    file.write("Linear MAE : {:.2f}  Neural MAE :{:.2f}\n".format(linear_mae,neural_mae))
    file.write("Linear MAPE : {:.2f}  Neural MAPE :{:.2f}\n\n".format(linear_mape, neural_mape))
    if linear_mae < neural_mae:
        linear_count = linear_count + 1
        system_list.append(entry)
    else:
        neural_count = neural_count + 1

print("Linear has {} wins".format(linear_count))
print("Neural has {} wins".format(neural_count))
file.write("Linear has {} wins\n".format(linear_count))
file.write("Neural has {} wins\n\n".format(neural_count))

print("The systems where linear beat neural were {}".format(system_list))
file.write("The systems where linear beat neural were {}\n\n".format(system_list))
for system in systems:
    print(system)
    print("Linear Mean MAE is {:.2f}".format(linear_errors[system] / systems_length[system]))
    print("Neural Mean MAE is {:.2f}".format(neural_errors[system] / systems_length[system]))
    print("Linear Mean MAPE is {:.2f}".format(linear_cents[system] / systems_length[system]))
    print("Neural Mean MAPE is {:.2f}\n".format(neural_cents[system] / systems_length[system]))
    file.write(system)
    file.write("\n")
    file.write("Linear Mean MAE is {:.2f}\n".format(linear_errors[system] / systems_length[system]))
    file.write("Neural Mean MAE is {:.2f}\n".format(neural_errors[system] / systems_length[system]))
    file.write("Linear Mean MAPE is {:.2f}\n".format(linear_cents[system] / systems_length[system]))
    file.write("Neural Mean MAPE is {:.2f}\n\n".format(neural_cents[system] / systems_length[system]))

stat =  wilcoxon(linear_cents_array,neural_cents_array)
print(stat)
print("The median of neural is : {}".format(np.median(neural_cents_array)))
print("The median of linear is : {}".format(np.median(linear_cents_array)))


file.write(str(stat))
file.write("\n")
file.write("The median of neural is : {}\n".format(np.median(neural_cents_array)))
file.write("The median of linear is : {}".format(np.median(linear_cents_array)))
file.close()