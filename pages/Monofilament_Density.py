import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_monofilament_hex(ax, density, diameter, title):
    area_each = np.pi * (diameter / 2) ** 2
    total_area = density * area_each
    percent_coverage = (total_area / 1.0) * 100

    n_total = density * 1.0
    n_rows = int(np.sqrt(n_total / (np.sqrt(3)/2)))
    row_spacing = 1.0 / n_rows
    col_spacing = row_spacing * np.sqrt(3)/2

    y = 0
    row_num = 0
    while y < 1.0:
        x_offset = 0 if row_num % 2 == 0 else col_spacing / 2
        x = x_offset
        while x < 1.0:
            circle = patches.Circle((x, y), diameter / 2, edgecolor='black', facecolor='gray', lw=0.2)
            ax.add_patch(circle)
            x += col_spacing
        y += row_spacing
        row_num += 1

    ax.plot([0,1,1,0,0], [0,0,1,1,0], 'k-', lw=1)
    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("inches")
    ax.set_title(f"{title}\n{density} ends/in², {diameter}\" dia")
    ax.text(0.5, -0.12, f"Monofilament area = {total_area:.4f} in²\nCoverage = {percent_coverage:.1f}%", 
            transform=ax.transAxes, ha='center', fontsize=9)

def draw_monofilament_grid(ax, density, diameter, title):
    area_each = np.pi * (diameter / 2) ** 2
    total_area = density * area_each
    percent_coverage = (total_area / 1.0) * 100

    n_filaments = int(np.sqrt(density))
    spacing = 1.0 / n_filaments

    for i in range(n_filaments):
        for j in range(n_filaments):
            x = (i + 0.5) * spacing
            y = (j + 0.5) * spacing
            circle = patches.Circle((x, y), diameter / 2, edgecolor='black', facecolor='gray', lw=0.2)
            ax.add_patch(circle)

    ax.plot([0,1,1,0,0], [0,0,1,1,0], 'k-', lw=1)
    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("inches")
    ax.set_title(f"{title}\n{density} ends/in², {diameter}\" dia")
    ax.text(0.5, -0.12, f"Monofilament area = {total_area:.4f} in²\nCoverage = {percent_coverage:.1f}%", 
            transform=ax.transAxes, ha='center', fontsize=9)

# Streamlit UI
st.title("Monofilament Pattern Visualizer")
st.write("Visual representation of different brush configurations using hex and grid patterns.")

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

draw_monofilament_hex(axs[0], 6912, 0.006, "AngleOn™")
draw_monofilament_grid(axs[1], 2273, 0.010, "XT10")
draw_monofilament_grid(axs[2], 1136, 0.016, "XT16")
draw_monofilament_hex(axs[3], 9754, 0.0045, "Brushlon")

st.pyplot(fig)
