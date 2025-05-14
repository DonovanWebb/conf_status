import matplotlib.pyplot as plt
import numpy as np
import graph_style
from scipy.optimize import curve_fit

graph_style.set_graph_style()


date = "2025-05-13"
rid = 26744

data_dict = graph_style.load_data(rid, date)
print(data_dict["datasets"].keys())
num_cliff = np.array(data_dict["datasets"]["data.rbm.num_cliffords"])
survivals = np.array(data_dict["datasets"]["data.rbm.survivals"])
survivals_err = np.array(data_dict["datasets"]["data.rbm.survival_errs"])
outcomes = data_dict["datasets"]["data.circuits.outcomes"]
sequences = data_dict["datasets"]["data.circuits.sequences"]


def get_seq_length(seq):
    # convert bytes-like object to string
    q_sequence = seq.decode("utf-8")
    # split the string into a list of strings
    q_sequence = q_sequence.split(";")
    # count the number of cliffords in the sequence
    return len(q_sequence)


seq_lens = np.array([get_seq_length(seq) for seq in sequences])


def seq_to_cliffs(seq_lens):
    offs = 2
    max_len = np.max(num_cliff)
    n_lens = np.unique(num_cliff).shape[0] - 1
    gate_per_cliff = 3
    seq_cliff_lens = (
        np.round((seq_lens / (gate_per_cliff * (max_len / n_lens))), 0)
        * ((max_len - offs) / (n_lens))
        + offs
    ) // 1

    return seq_cliff_lens


seq_cliff_lens = seq_to_cliffs(seq_lens)

cliff100 = seq_cliff_lens == 100.0
surv100 = outcomes[cliff100]
print(surv100)  ## From here compare to see spread of outcomes


arg_sort = np.argsort(num_cliff)
num_cliff = num_cliff[arg_sort]
survivals = survivals[arg_sort]
survivals_err = survivals_err[arg_sort]

# Fit linear function to the data:
popt, pcov = curve_fit(graph_style.linear_func, num_cliff, survivals, p0=[-1e-3, 1])
print("Slope: ", popt[0])
print("Intercept: ", popt[1])
plt.errorbar(num_cliff, survivals, yerr=survivals_err, fmt="x")
plt.plot(
    num_cliff, graph_style.linear_func(num_cliff, popt[0], popt[1]), label="Linear fit"
)
plt.xlabel("Number of Cliffords")
plt.ylabel("Average Sequence Fidelity")
# plt.show()
