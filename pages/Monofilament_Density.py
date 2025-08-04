import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

@st.cache_resource
def generate_monofilament_data(density, diameter, pattern="hex"):
    # Compute filament cross-sectional area and total area
    area_each = np.pi * (diameter / 2) ** 2
    total_area = density * area_each
    percent_coverage = (total_area / 1.0) * 100

    positions = []

    if pattern == "hex":
        n_total = density * 1.0
        n_rows = int(np.sqrt(n_total / (np.sqrt(3) / 2)))
        row_spacing = 1.0 / n_rows
        col_spacing = row_spacing * np.sqrt(3) / 2

        y = 0
        row_num = 0
        while y < 1.0:
            x_offset = 0 if row_num % 2 == 0 else col_spacing / 2
            x = x_offset
            while x < 1.0:
                positions.append((x, y))
                x += col_spacing
            y += row_spacing
            row_num += 1

    elif pattern == "grid":
        n_filaments = int(np.sqrt(density))
        spacing = 1.0 / n_filaments
        for i in range(n_filaments):
            for j in range(n_filaments):
                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing
                positions.append((x, y))

    return np.array(positions), total_area, percent_coverage

def draw_monofilament(ax, positions, diameter, title, density, total_area, percent_coverage):
    if len(positions) > 0:
        ax.scatter(positions[:, 0], positions[:, 1], s=(diameter * 72)**2, edgecolor='black', facecolor='gray', linewidth=0.2)

    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', lw=1)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("inches")
    ax.set_title(f"{title}\n{density} ends/in², {diameter}\" dia")
    ax.text(0.5, -0.12, f"Monofilament area = {total_area:.4f} in²\nCoverage = {percent_coverage:.1f}%", 
            transform=ax.transAxes, ha='center', fontsize=9)

# Setup plot
fig, axs = plt.subplots(1, 4, figsize=(20, 5), dpi=150)

# Data for each brush type
configs = [
    ("AngleOn™", 6912, 0.006, "hex"),
    ("XT10", 2273, 0.010, "grid"),
    ("XT16", 1136, 0.016, "grid"),
    ("Competitor", 9754, 0.0045, "hex"),
]

for ax, (title, density, diameter, pattern) in zip(axs, configs):
    positions, total_area, percent_coverage = generate_monofilament_data(density, diameter, pattern)
    draw_monofilament(ax, positions, diameter, title, density, total_area, percent_coverage)

plt.tight_layout()
st.pyplot(fig)
