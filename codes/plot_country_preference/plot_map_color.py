import matplotlib.pyplot as plt
import numpy as np

from geonamescache import GeonamesCache
from geonamescache.mappers import country
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap

import json
import pandas as pd

pref_type = "neutral"

country_color = json.load(open(f"../../results/threshold_0.6_filter_True/data/year_1990/gender_{pref_type}_country_bf10.json"))
shapefile = "ne_10m_admin_0_countries/ne_10m_admin_0_countries"

cmap = {1: "#6E99DD", 2: "#FF5072", 0: "#3ac8a4"}
df = pd.DataFrame()
df["country"] = country_color.keys()
df["color"] = [cmap[item] for item in country_color.values()]
mapper = country(from_key='name', to_key='iso3')
df["country_code"] = [mapper(cnt) for cnt in df["country"]]
df = df.set_index("country_code")

gc = GeonamesCache()
iso3_codes = list(gc.get_dataset_by_key(gc.get_countries(), 'iso3').keys())

fig = plt.figure(figsize=(22, 12))

ax = fig.add_subplot(111, frame_on=False)

m = Basemap(lon_0=0, projection='robin')
m.drawmapboundary(color='w')

m.readshapefile(shapefile, 'units', color='#444444', linewidth=.2)
for info, shape in zip(m.units_info, m.units):
    iso3 = info['ADM0_A3']
    if iso3 not in df.index:
        color = '#dddddd'
    else:
        color = df.loc[iso3]['color']
    patches = [Polygon(np.array(shape), True)]
    pc = PatchCollection(patches)
    pc.set_facecolor(color)
    ax.add_collection(pc)

# Cover up Antarctica so legend can be placed over it.
ax.axhspan(0, 1000 * 1800, facecolor='w', edgecolor='w', zorder=2)
plt.savefig("country_color_map_%s.svg"%pref_type, bbox_inches='tight')
# plt.show()


# # Set the map footer.
# plt.annotate(description, xy=(-.8, -3.2), size=14, xycoords='axes fraction')

# plt.savefig(imgfile, bbox_inches='tight', pad_inches=.2)
