import seaborn as sns
sns.set_context("paper", font_scale=1.2, rc={"lines.linewidth": 2.5})
colors = ["#6E99DD", "#FF5072", "#3ac8a4", "#E66100", "#5D3A9B", "#000000"]
sns.set_palette(sns.color_palette(colors))
darker_colors = ["#ADC4EA", "#FFA7B8", "#9CE3D1"]  # balance
lighter_colors = ["#cddbf2", "#ffd3db", "#cdf1e8"]  # neutral