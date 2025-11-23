#%%
import xarray as xr
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
import calendar
import pyinterp
import pyinterp.backends.xarray
import pandas as pd
import time
import dask.array as da
from tqdm import tqdm

def plot_side_by_side(ds1, ds2, x_dim='longitude', y_dim='latitude', title1='Dataset 1', title2='Dataset 2', cmap='turbo', width=500, height=400, display=False):
	"""
	Plots two xarray DataArrays side-by-side, automatically matching their
	resolutions and providing a shared HTML table that updates on hover to
	compare values. Zooming and panning are synchronized.
	
	Parameters:
	- ds1, ds2: xarray.DataArray - The two DataArrays to plot.
	- x_dim, y_dim: str - The names of the x and y dimensions in the datasets.
	- title1, title2: str - Titles for each plot.
	- cmap: str - Colormap to use for both plots.
	- width, height: int - Dimensions for each plot.
	- display: bool - If True, automatically opens the plot in a new tab.
	
	Returns:
	- A Panel layout object containing the interactive plots and table.
	"""
	# 1. Import libraries and set up extensions
	import hvplot.xarray
	import holoviews as hv
	from holoviews import streams
	import panel as pn
	import numpy as np
	hv.extension('bokeh')
	pn.extension()

	# 2. Match grid resolutions before plotting
	if ds1.size < ds2.size:
		print(f"Notice: Reindexing '{title1}' to match the resolution of '{title2}'.")
		ds1 = ds1.reindex_like(ds2, method='nearest')
	elif ds2.size < ds1.size:
		print(f"Notice: Reindexing '{title2}' to match the resolution of '{title1}'.")
		ds2 = ds2.reindex_like(ds1, method='nearest')

	# 3. Create the two plots using the specified dimension names
	plot1 = ds1.hvplot.quadmesh(
		x=x_dim, y=y_dim, rasterize=False,
		cmap=cmap, title=title1, width=width, height=height,
		colorbar=True, clim=(float(ds1.min()), float(ds1.max())),
		tools=['hover', 'wheel_zoom', 'pan']
	)

	plot2 = ds2.hvplot.quadmesh(
		x=x_dim, y=y_dim, rasterize=False,
		cmap=cmap, title=title2, width=width, height=height,
		colorbar=True, clim=(float(ds2.min()), float(ds2.max())),
		tools=['hover', 'wheel_zoom', 'pan']
	)

	# 4. Set up streams to capture hover position from either plot
	pointer1 = streams.PointerXY(source=plot1, x=np.nan, y=np.nan)
	pointer2 = streams.PointerXY(source=plot2, x=np.nan, y=np.nan)

	# 5. Define a function to create the HTML table
	def create_html_table(x, y):
		val1_str, val2_str = 'N/A', 'N/A'
		try:
			if x is not None and y is not None and not np.isnan(x) and not np.isnan(y):
				# UPDATED: Interpolate using a dictionary with the dynamic dim names
				interp_coords = {x_dim: x, y_dim: y}
				val1 = ds1.interp(interp_coords, method='nearest').values
				val2 = ds2.interp(interp_coords, method='nearest').values
				
				# UPDATED: Explicitly check for NaN before formatting
				if not np.isnan(val1):
					val1_str = f"{val1:.4f}"
				if not np.isnan(val2):
					val2_str = f"{val2:.4f}"
		except (ValueError, IndexError, KeyError):
			# Catch errors if coords are out of bounds or names are wrong
			pass
		
		lon_str = f"{x:.2f}" if x is not None and not np.isnan(x) else "N/A"
		lat_str = f"{y:.2f}" if y is not None and not np.isnan(y) else "N/A"
		
		return f"""
		<table border="1" style="width: 100%; text-align: center; border-collapse: collapse; font-family: sans-serif;">
			<tr style="background-color: #f2f2f2;">
				<th style="padding: 8px;">{title1}</th>
				<th style="padding: 8px;">{title2}</th>
			</tr>
			<tr>
				<td style="padding: 8px; font-size: 1.1em;"><b>{val1_str}</b></td>
				<td style="padding: 8px; font-size: 1.1em;"><b>{val2_str}</b></td>
			</tr>
			<tr>
				<td colspan="2" style="padding: 8px; font-size: 0.9em; color: #555;">
					{x_dim.capitalize()}: {lon_str}, {y_dim.capitalize()}: {lat_str}
				</td>
			</tr>
		</table>
		"""

	# 6. Create the Panel pane for the HTML table
	html_pane = pn.pane.HTML(create_html_table(np.nan, np.nan), width=width*2)

	# 7. Define the function that updates the table when a hover event occurs
	def update_table_on_hover(*events):
		active_x = pointer1.x if not np.isnan(pointer1.x) else pointer2.x
		active_y = pointer1.y if not np.isnan(pointer1.y) else pointer2.y
		html_pane.object = create_html_table(active_x, active_y)

	# 8. Watch both pointers for changes and link them to the update function
	pointer1.param.watch(update_table_on_hover, ['x', 'y'])
	pointer2.param.watch(update_table_on_hover, ['x', 'y'])

	# 9. Arrange the plots and link their axes
	side_by_side_plots = plot1 + plot2

	# 10. Combine the plots and the table into a final layout
	full_layout = pn.Column(side_by_side_plots, html_pane)
	
	if display:
		full_layout.show()

	return full_layout

def plot_interactive_timeseries(
    da, 
    x_dim='longitude', 
    y_dim='latitude', 
    time_dim='time', 
    cmap='turbo', 
    width=600, 
    height=500,
    rasterize=True,
    display=False):
    # 1. Use .hvplot() with the 'groupby' argument
    # The 'groupby' parameter automatically creates a widget (slider) for the specified dimension.
	import xarray as xr
	import numpy as np
	import hvplot.xarray
	import holoviews as hv
	import panel as pn

	# Set the HoloViews backend and Panel extension
	hv.extension('bokeh')
	pn.extension()

	interactive_plot = da.hvplot.quadmesh(
		x=x_dim,
		y=y_dim,
		groupby=time_dim,  # This creates the time slider
		rasterize=rasterize,
		cmap=cmap,
		width=width,
		height=height,
		colorbar=True,
		title=f"{da.name or 'Data'} over {time_dim.capitalize()}",
		dynamic=False  # Pre-computes all frames for a smooth slider experience
			)

	# if display:
	# 	# To display in a script or new tab, wrap it in a Panel layout and call .show()
	# 	pn.Column(interactive_plot).show()

	return interactive_plot

input_dir=Path('/scratch/depfg/7006713/temp/1km_forcing/input')
save_folder=Path('/scratch/depfg/7006713/temp/1km_forcing/output')


downscaled = xr.open_zarr(save_folder / 'evap.zarr', chunks={})['evap'].isel(time=slice(0, 731))
downscaled = downscaled.mean('time').compute()

lat_min = float(downscaled.latitude.min())
lat_max = float(downscaled.latitude.max())
lon_min = float(downscaled.longitude.min())
lon_max = float(downscaled.longitude.max())

# crop original to the spatial bounds of the downscaled product
orig_ds = xr.open_dataset(input_dir / 'forcing/W5E5/refPotEvap_W5E5v2.0_1979_2019.nc')['PM_FAO_56']
orig_ds = orig_ds.isel(time=slice(0, 731)).rename({'lat':'latitude', 'lon':'longitude'})
orig_ds = orig_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
orig_ds = orig_ds.mean('time').compute()
print(orig_ds)
print(downscaled)
# plot_interactive_timeseries(downscaled) # Single map with time slider
plot_side_by_side(orig_ds, downscaled, title1='Original', title2='Downscaled')
# %%