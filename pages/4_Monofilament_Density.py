import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

@st.cache_resource
def generate_monofilament_data(density, diameter, pattern="hex"):
    area_each = np.pi * (diameter / 2) ** 2
    total_area = density * area_each
    percent_coverage = total_area * 100

    positions = []

    if pattern == "hex":
        n_total = density
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

def draw_monofilament(positions, diameter, title, density, total_area, percent_coverage):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    if len(positions) > 0:
        ax.scatter(positions[:, 0], positions[:, 1], 
                   s=(diameter * 72)**2, edgecolor='black', facecolors='gray', linewidth=0.2)

    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', lw=1)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("inches")
    ax.set_ylabel("inches")
    ax.set_title(f"{title}\n{density} ends/in², {diameter}\" dia")
    ax.text(0.5, -0.12, f"Monofilament area = {total_area:.4f} in²\nCoverage = {percent_coverage:.1f}%", 
            transform=ax.transAxes, ha='center', fontsize=9)
    plt.tight_layout()
    return fig

# Streamlit UI
st.set_page_config(page_title="Monofilament Pattern Viewer", layout="centered")
st.title("Monofilament Pattern Visualizer")

# Inputs
pattern = st.selectbox("Pattern Type", ["hex", "grid"])
density = st.slider("Filament Density (ends/in²)", 1000, 12000, 6912, step=100)
diameter = st.slider("Filament Diameter (in)", 0.002, 0.02, 0.006, step=0.0005)

positions, total_area, percent_coverage = generate_monofilament_data(density, diameter, pattern)
fig = draw_monofilament(positions, diameter, "Custom Pattern", density, total_area, percent_coverage)
st.pyplot(fig)
