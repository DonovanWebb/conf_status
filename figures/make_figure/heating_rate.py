import matplotlib.pyplot as plt
import numpy as np
import graph_style
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter


graph_style.set_graph_style()

date = "2024-12-13"
rid = 11740

data_dict = graph_style.load_data(rid, date)
# for x in data_dict["datasets"].keys():
# print(x)

delays0 = np.array(data_dict["datasets"][f"ndscan.rid_{rid}.points.axis_0"]) * 1000
freqs = np.array(data_dict["datasets"][f"ndscan.rid_{rid}.points.axis_1"])
flop_durations = np.array(
    data_dict["datasets"][f"ndscan.rid_{rid}.points.channel__axis_0"]
)
nbar = np.array(data_dict["datasets"][f"ndscan.rid_{rid}.points.channel__nbar"])
nbar_err = np.array(data_dict["datasets"][f"ndscan.rid_{rid}.points.channel__nbar_err"])
np.array(data_dict["datasets"][f"ndscan.rid_{rid}.points.channel__spec"])

# Chose all entries with delays1 = f_ax
f_ax = 1.07
delays0 = delays0[freqs == f_ax]
nbar = nbar[freqs == f_ax]
nbar_err = nbar_err[freqs == f_ax]

p_fit, p_cov = curve_fit(graph_style.linear_func, delays0, nbar)

ndot = p_fit[0] * 1000
ndot_err = np.sqrt(p_cov[0][0]) * 1000
print(ndot)
print(ndot_err)

plt.errorbar(delays0, nbar, yerr=nbar_err, fmt="o")
plt.plot(delays0, graph_style.linear_func(delays0, *p_fit), label="linear heating fit")
plt.xlabel("Delay (ms)")
plt.ylabel("$\\bar{n}$")

plt.savefig("heating_rate.png")
