import matplotlib.pyplot as plt
import numpy as np
import graph_style
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter


def make_data(times, sideband_idx):
    """Stack together probe durations with sideband index for ..."""
    return np.vstack((times, np.full(len(times), sideband_idx)))


def thermal_sideband_flop(data, eta_omega, n_bar, a, c, n_max=100):
    """Calculate the population for a square probe pulse of the given duration on a
    (plus/minus) first motional sideband.

    :param data: An array of shape (2, n), where the first axis enumerates a tuple of
        pulse duration and sign of the sideband to consider (-1 for RSB, 1 for BSB),
        and the calculation is broadcast across the second axis.
    :param eta_omega: The effective sideband Rabi frequency.
    :param n_bar: The mean occupation number of the thermal state.
    :param a: Fudge factor for the amplitude of the oscillations.
    :param c: Fudge factor adding a constant vertical offset to the oscillations.
    :param n_max: The number of Fock states to consider in the calculation. Should be
        large enough (i.e. much larger than n_bar) so there is negligible population in
        the trucated states.

    :return: An array of shape (n,), giving the resulting populations (starting at 1).
    """
    t, sideband_sign = data

    ns = np.arange(n_max)
    omega_eff = eta_omega * (ns + (1 + sideband_sign[:, np.newaxis]) / 2) ** 0.5
    pop = 0.5 * (1 - np.cos(omega_eff * t[:, np.newaxis]))

    weights = (n_bar / (1 + n_bar)) ** ns / (1 + n_bar)
    weights /= np.sum(weights)

    return a * (1 - np.sum(pop * weights, axis=1)) + c


def thermal_dist(n_bar, n_max=100):
    ns = np.arange(n_max)
    weights = (n_bar / (1 + n_bar)) ** ns / (1 + n_bar)
    weights /= np.sum(weights)
    return weights


graph_style.set_graph_style()

date = "2025-04-29"
rid = 22922

data_dict = graph_style.load_data(rid, date)
# for x in data_dict["datasets"].keys():
#   print(x)

duration = np.array(data_dict["datasets"][f"ndscan.rid_{rid}.point._axis_0"]) * 1e6
y_b = np.array(
    data_dict["datasets"][
        f"ndscan.rid_{rid}.point._channel_readout_pi_rsb_pi_measurement_1_p"
    ]
)
y_b_err = np.array(
    data_dict["datasets"][
        f"ndscan.rid_{rid}.point._channel_readout_pi_rsb_pi_measurement_1_p_err"
    ]
)
y_r = np.array(
    data_dict["datasets"][
        f"ndscan.rid_{rid}.point._channel_readout_rsb_measurement_1_p"
    ]
)
y_r_err = np.array(
    data_dict["datasets"][
        f"ndscan.rid_{rid}.point._channel_readout_rsb_measurement_1_p_err"
    ]
)
eta_omega = np.array(data_dict["datasets"][f"ndscan.rid_{rid}.point._eta_omega"])
eta_omega_err = np.array(
    data_dict["datasets"][f"ndscan.rid_{rid}.point._eta_omega_err"]
)
nbar = np.array(data_dict["datasets"][f"ndscan.rid_{rid}.point._nbar"])
nbar_err = np.array(data_dict["datasets"][f"ndscan.rid_{rid}.point._nbar_err"])

sort_index = np.argsort(duration)
duration = duration[sort_index]
y_b = y_b[sort_index]
y_b_err = y_b_err[sort_index]
y_r = y_r[sort_index]
y_r_err = y_r_err[sort_index]


x = np.hstack((make_data(duration, 1), make_data(duration, -1)))
y = np.hstack((y_b, y_r))
y_err = np.hstack((y_b_err, y_r_err))
a0 = 1
c0 = 0
t_pi = 80

p_fit, p_cov = curve_fit(
    thermal_sideband_flop,
    x,
    y,
    p0=[2 * np.pi / t_pi, 0.1, a0, c0],
    sigma=y_err,
    absolute_sigma=True,
)

n_bar = p_fit[1]

fock_state = thermal_dist(n_bar, n_max=10)

fig_width = graph_style.get_fig_width()
fig_height = 1.5 * fig_width / graph_style.get_phi()
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(fig_width, fig_height))

linspace = np.linspace(0, np.max(duration), 1000)
ax[0].plot(
    linspace,
    thermal_sideband_flop(make_data(linspace, 1), *p_fit),
    label="Thermal $\pi$-RSB-$\pi$ fit",
)

ax[0].errorbar(
    duration,
    y_b,
    yerr=y_b_err,
    fmt="o",
    label="pi_rsb_pi_p",
)
ax[0].set_ylabel("Population $\pi$-RSB-$\pi$")

ax[1].plot(
    linspace,
    thermal_sideband_flop(make_data(linspace, -1), *p_fit),
    label="Thermal RSB fit",
)

ax[1].errorbar(
    duration,
    y_r,
    yerr=y_r_err,
    fmt="o",
    label="rsb_p",
)
ax[1].set_xlabel("Duration ($\mu$s)")
ax[1].set_ylabel("Population RSB")

ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
plt.savefig("sideband_thermometry.pdf")

fig1, ax1 = plt.subplots(1, 1)
ax1.bar(
    np.arange(len(fock_state)),
    fock_state,
    width=0.5,
    alpha=0.7,
)
# ax1.set_yscale("log")
plt.savefig("fock_state_distribution.pdf")
