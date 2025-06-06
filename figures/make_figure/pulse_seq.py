"""
A pulse sequence figure for state preparation of a qubit.
This state preparation uses pi-pulses on the state m+1/2 to m-3/2
followed by deshelving pulses with the 854 laser.
This is repeated N times
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Settings
block_height = 1
block_width = 1.5
spacing = 0.1
N = 3  # Number of repeats

# Sequence to repeat
repeat_seq = [
    ("π-pulse", "gray"),
    ("deshelve\n854", "lightgray"),
]

# Set up figure
fig, ax = plt.subplots(figsize=(10, 2))
x = 0


# Draw a labeled block
def draw_block(label, color, x_pos):
    rect = patches.Rectangle(
        (x_pos, 0),
        block_width,
        block_height,
        edgecolor="black",
        facecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(
        x_pos + block_width / 2,
        block_height / 2,
        label,
        ha="center",
        va="center",
        fontsize=10,
    )


# Draw N repeats
for i in range(N):
    for label, color in repeat_seq:
        draw_block(label, color, x)
        x += block_width + spacing

# Draw time axis
ax.plot([0, x], [-0.2, -0.2], color="black", lw=1)
ax.annotate("", xy=(x, -0.2), xytext=(0, -0.2), arrowprops=dict(arrowstyle="->"))

# Add "N repeats" label centered below the repeated sequence
center_x = x / 2
ax.text(center_x, -0.5, f"{N} repeats", ha="center", va="top", fontsize=11)

# Formatting
ax.axis("off")
ax.set_xlim(0, x + 0.5)
ax.set_ylim(-1, 1.8)

plt.tight_layout()
plt.show()
