from pathlib import Path
import xarray as xr
import numpy as np
from dataclasses import dataclass, field
import calendar
import pyinterp
import pyinterp.backends.xarray
import pandas as pd
import time
import dask.array as da
from tqdm import tqdm

def get_adjusted_day_of_year(time_step: np.datetime64) -> int:
	# Convert to string in 'YYYY-MM-DD' format
	curr_time_step_str = np.datetime_as_string(time_step, unit='D')

	# Convert back to np.datetime64, then to datetime object
	curr_time_step = np.datetime64(curr_time_step_str).astype('datetime64[D]').astype(object)

	# Get day of year
	doy_step = curr_time_step.timetuple().tm_yday

	# Adjust for leap year if DOY > 59
	if calendar.isleap(curr_time_step.year) and doy_step > 59:
		doy_step -= 1

	return doy_step

def get_month_of_year(time_step: np.datetime64) -> int:
	pd_timestamp = pd.to_datetime(time_step)
	return pd_timestamp.month

@dataclass
class Config:
	coords: dict    
	input_dir: str
	mask_ds: str 
	tas_ds: str
	tas_cf: str
	evap_ds: str
	evap_cf: str
	tp_ds: str
	tp_cf: str
	tp_drizzle: str
	save_folder: str
	chunks: dict
 
	@staticmethod
	def pad_with_extrapolated_coords(ds, dims=['longitude', 'latitude']):
		# Start with the original dataset
		padded_ds = ds
		
		# Loop through each dimension specified in the list
		for dim in dims:
			# Pad the entire object using the standard 'edge' method for this dimension
			# This correctly pads the data variables.
			temp_ds = padded_ds.pad({dim: 1}, mode='edge')

			# Now, create the new, extrapolated coordinates to replace the old ones
			original_coords = padded_ds.coords[dim].values
			
			# Calculate the step differences at both ends of the coordinate array
			start_step = original_coords[1] - original_coords[0]
			end_step = original_coords[-1] - original_coords[-2]

			# Create the new, extrapolated coordinate array
			new_coords = np.empty(len(original_coords) + 2, dtype=original_coords.dtype)
			new_coords[0] = original_coords[0] - start_step  # Extrapolate at the start
			new_coords[-1] = original_coords[-1] + end_step # Extrapolate at the end
			new_coords[1:-1] = original_coords

			# Assign the new, correct coordinates to the temporarily padded dataset
			# and update the main padded_ds for the next loop iteration.
			padded_ds = temp_ds.assign_coords({dim: new_coords})

		return padded_ds
	
	def __post_init__(self):
		"""Automatically process attributes after initialization."""
		self.input_dir = Path(self.input_dir)
		
		def load_tas_data(input_dir, tas_ds, tas_cf, coords, chunks):
			# Load tas_ds
			ds_tas = xr.open_dataset(input_dir / tas_ds, engine='h5netcdf', chunks={'time': 1, 'latitude': -1, 'longitude': -1})['tas']
			ds_tas.attrs = {}
			ds_tas = ds_tas.rename({'lat': 'latitude', 'lon': 'longitude'})
			ds_tas = ds_tas.sel(**coords).sortby('latitude').transpose('longitude', 'latitude', 'time')
			ds_tas = self.pad_with_extrapolated_coords(ds_tas)

			# Load tas_cf
			ds_cf = xr.open_zarr(input_dir / tas_cf, chunks={'time': 1, 'latitude': -1, 'longitude': -1})['ta_correction_factor']
			ds_cf = ds_cf.rename({'lat': 'latitude', 'lon': 'longitude'})
			ds_cf = ds_cf.sel(**coords).sortby('latitude').transpose('longitude', 'latitude', 'time')
			n_time = ds_cf.sizes['time']
			ds_cf = ds_cf.assign_coords(time=np.arange(1, n_time + 1)).rename('tas_cf')
			return ds_tas, ds_cf

	
		def load_evap_data(input_dir, evap_ds, evap_cf, coords, chunks):
			# Load evap_ds
			ds_evap = xr.open_dataset(input_dir / evap_ds, engine='h5netcdf', chunks={'time': 1, 'latitude': -1, 'longitude': -1})['PM_FAO_56']
			ds_evap = ds_evap.rename({'lat': 'latitude', 'lon': 'longitude'})
			ds_evap = ds_evap.sel(**coords).sortby('latitude').transpose('longitude', 'latitude', 'time').rename('evap')
			ds_evap = self.pad_with_extrapolated_coords(ds_evap)

			# Load evap_cf
			ds_cf = xr.open_zarr(input_dir / evap_cf, chunks={'time': 1, 'latitude': -1, 'longitude': -1})['pet_correction_factor']
			ds_cf = ds_cf.rename({'lat': 'latitude', 'lon': 'longitude'})
			ds_cf = ds_cf.sel(**coords).sortby('latitude').transpose('longitude', 'latitude', 'time')
			n_time = ds_cf.sizes['time']
			ds_cf = ds_cf.assign_coords(time=np.arange(1, n_time + 1)).rename('evap_cf')

			return ds_evap, ds_cf

		
		def load_tp_data(input_dir, tp_ds, tp_cf, tp_drizzle, coords, chunks):
			ds_tp = xr.open_dataset(input_dir / tp_ds, engine='h5netcdf', chunks={'time': 1, 'latitude': -1, 'longitude': -1})['pr']
			ds_tp = ds_tp.rename({'lat': 'latitude', 'lon': 'longitude'})
			ds_tp = ds_tp.sel(**coords).sortby('latitude').transpose('longitude', 'latitude', 'time').rename('tp')
			ds_tp.attrs = {}
			ds_tp = self.pad_with_extrapolated_coords(ds_tp)

			# Load tp_cf
			ds_cf = xr.open_zarr(input_dir / tp_cf, chunks={'time': 1, 'latitude': -1, 'longitude': -1})['tp_correction_factor']
			ds_cf = ds_cf.rename({'lat': 'latitude', 'lon': 'longitude'})
			ds_cf = ds_cf.sel(**coords).sortby('latitude').transpose('longitude', 'latitude', 'time')
			n_time = ds_cf.sizes['time']
			ds_cf = ds_cf.assign_coords(time=np.arange(1, n_time + 1)).rename('tp_cf')

			# Load tp_drizzle
			ds_drizzle = xr.open_zarr(input_dir / tp_drizzle, chunks={'time': 1, 'latitude': -1, 'longitude': -1})['dry_days_cutoff']
			ds_drizzle = ds_drizzle.rename({'lat': 'latitude', 'lon': 'longitude'})
			ds_drizzle = ds_drizzle.sel(**coords).sortby('latitude').transpose('longitude', 'latitude', 'time')
			n_time = ds_drizzle.sizes['time']
			ds_drizzle = ds_drizzle.assign_coords(time=np.arange(1, n_time + 1)).rename('tp_drizzle')

			return ds_tp, ds_cf, ds_drizzle

		def load_mask(input_dir, mask_ds, cf_ds, coords, chunks):
			grid = xr.open_dataset(input_dir / mask_ds, chunks={}).rename({'lat': 'latitude', 'lon': 'longitude', 'gwStorage': 'mask'})['mask']
			grid = grid.sel(**coords).sortby('latitude').transpose('longitude', 'latitude')
			grid = grid.reindex(latitude=cf_ds.latitude.values, longitude=cf_ds.longitude.values, method='nearest')
			grid = xr.where(grid.notnull(), 1.0, np.nan).chunk({'latitude': chunks['latitude'], 'longitude': chunks['longitude']})
			return grid.persist()
		
		def create_output_datasets(tas_ds, tas_cf, save_folder, chunks):
			# Create a new DataArray with the time dimension from tas_ds and lat/lon from tas_cf
			ds_template = xr.DataArray(
				da.full(
					(tas_cf.latitude.size, tas_cf.longitude.size, tas_ds.time.size),
					np.nan,
					dtype=np.float32,
					chunks=(tas_cf.latitude.size, tas_cf.longitude.size, 1)
				),
				dims=['latitude', 'longitude', 'time'],
				coords={
					'latitude': tas_cf.latitude,
					'longitude': tas_cf.longitude,
					'time': tas_ds.time
				},
			)
   
			Path(save_folder).mkdir(parents=True, exist_ok=True)
			for var in ['tas', 'tp', 'evap']:
				ds = ds_template.to_dataset(name=f'{var}')
				for v in ds.data_vars:
					ds[v].encoding['chunks'] = (chunks['latitude'], chunks['longitude'], chunks['time'])
				ds.to_zarr(Path(save_folder) / f'{var}.zarr', mode = 'w', compute=False, consolidated=True, zarr_format=2)
			return Path(save_folder)

		self.tas_ds, self.tas_cf = load_tas_data(self.input_dir, self.tas_ds, self.tas_cf, self.coords, self.chunks)
		self.evap_ds, self.evap_cf = load_evap_data(self.input_dir, self.evap_ds, self.evap_cf, self.coords, self.chunks)
		self.tp_ds, self.tp_cf, self.tp_drizzle = load_tp_data(self.input_dir, self.tp_ds, self.tp_cf, self.tp_drizzle, self.coords, self.chunks)
		self.mask_ds = load_mask(self.input_dir, self.mask_ds, self.tas_cf, self.coords, self.chunks)
		
		# # HACK 
		self.tas_ds = self.tas_ds.sel(time=slice('1990-01-01', '2019-12-31'))
		self.evap_ds = self.evap_ds.sel(time=slice('1990-01-01', '2019-12-31'))
		self.tp_ds = self.tp_ds.sel(time=slice('1990-01-01', '2019-12-31'))
  
		# self.tas_ds = self.tas_ds.sel(time=slice('1990-01-01', '1990-12-31'))
		# self.evap_ds = self.evap_ds.sel(time=slice('1990-01-01', '1990-12-31'))
		# self.tp_ds = self.tp_ds.sel(time=slice('1990-01-01', '1990-12-31'))

		# self.tas_ds = self.tas_ds.sel(time=slice('1990-01-01', '1990-01-31'))
		# self.evap_ds = self.evap_ds.sel(time=slice('1990-01-01', '1990-01-31'))
		# self.tp_ds = self.tp_ds.sel(time=slice('1990-01-01', '1990-01-31'))
		
  		# # HACK
  	
		self.save_folder = create_output_datasets(self.tas_ds, self.tas_cf, self.save_folder, self.chunks)	

def downscale_tas(config):
    tas_ds_sub_template = pyinterp.backends.xarray.Grid2D(
        config.tas_ds.isel(time=0).drop_vars('time'), 
        geodetic=False
    )
    mx, my = np.meshgrid(
        config.tas_cf.longitude.values, 
        config.tas_cf.latitude.values, 
        indexing="ij"
    )
    mx_ravel, my_ravel = mx.ravel(), my.ravel()
    
    # Template for batch processing
    _corrected_ds = xr.DataArray(
        mx, 
        dims=['longitude', 'latitude'],
        coords=dict(
            latitude=config.tas_cf.latitude.values,
            longitude=config.tas_cf.longitude.values
        )
    )
    
    time_values = config.tas_ds.time.values
    batch_size = 48
    
    # Process in batches
    for batch_start in tqdm(range(0, len(time_values), batch_size), desc="Downscaling tas (batches of 96)"):
        batch_end = min(batch_start + batch_size, len(time_values))
        batch_times = time_values[batch_start:batch_end]
        
        batch_results = []
        
        for time_step in batch_times:
            tas_ds_sub_template.array[:] = config.tas_ds.sel(time=time_step).values
            
            corrected_ds = _corrected_ds.copy()
            corrected_ds[:] = tas_ds_sub_template.bivariate(
                coords=dict(longitude=mx_ravel, latitude=my_ravel), 
                num_threads=2
            ).reshape(mx.shape)
            
            corrected_ds = corrected_ds.expand_dims(time=[time_step]).to_dataset(name='tas')
            corrected_ds = xr.where(
                config.mask_ds.notnull(), 
                corrected_ds, 
                np.nan
            ) + config.tas_cf.sel(time=get_adjusted_day_of_year(time_step))
            
            batch_results.append(corrected_ds)
        
        batch_ds = xr.concat(batch_results, dim='time')
        batch_ds = batch_ds.chunk(config.chunks)
        
        batch_ds.to_zarr(
            config.save_folder / 'tas.zarr', 
            mode='a', 
            region={
                "time": slice(batch_start, batch_end), 
                "latitude": slice(None, None), 
                "longitude": slice(None, None)
            }
        )


def downscale_evap(config):
    evap_ds_sub_template = pyinterp.backends.xarray.Grid2D(
        config.evap_ds.isel(time=0).drop_vars('time'), 
        geodetic=False
    )
    mx, my = np.meshgrid(
        config.evap_cf.longitude.values, 
        config.evap_cf.latitude.values, 
        indexing="ij"
    )
    mx_ravel, my_ravel = mx.ravel(), my.ravel()
    
    _corrected_ds = xr.DataArray(
        mx, 
        dims=['longitude', 'latitude'],
        coords=dict(
            latitude=config.evap_cf.latitude.values,
            longitude=config.evap_cf.longitude.values
        )
    )
    
    time_values = config.evap_ds.time.values
    batch_size = 48
    
    for batch_start in tqdm(range(0, len(time_values), batch_size), desc="Downscaling evap (batches of 96)"):
        batch_end = min(batch_start + batch_size, len(time_values))
        batch_times = time_values[batch_start:batch_end]
        
        batch_results = []
        
        for time_step in batch_times:
            evap_ds_sub_template.array[:] = config.evap_ds.sel(time=time_step).values
            
            corrected_evap = _corrected_ds.copy()
            corrected_evap[:] = evap_ds_sub_template.bivariate(
                coords=dict(longitude=mx_ravel, latitude=my_ravel), 
                num_threads=2
            ).reshape(mx.shape)
            
            corrected_evap = corrected_evap.expand_dims(time=[time_step]).to_dataset(name='evap')
            corrected_evap = xr.where(
                config.mask_ds.notnull(), 
                corrected_evap, 
                np.nan
            ) * config.evap_cf.sel(time=get_adjusted_day_of_year(time_step))
            corrected_evap = corrected_evap * 1000.0  # Convert from m/day to mm/day
            
            batch_results.append(corrected_evap)
        
        batch_ds = xr.concat(batch_results, dim='time')
        batch_ds = batch_ds.chunk(config.chunks)
        
        batch_ds.to_zarr(
            config.save_folder / 'evap.zarr', 
            mode='a', 
            region={
                "time": slice(batch_start, batch_end), 
                "latitude": slice(None, None), 
                "longitude": slice(None, None)
            }
        )


def downscale_tp(config):
    tp_ds_sub_template = pyinterp.backends.xarray.Grid2D(
        config.tp_ds.isel(time=0).drop_vars('time'), 
        geodetic=False
    )
    mx, my = np.meshgrid(
        config.tp_cf.longitude.values, 
        config.tp_cf.latitude.values, 
        indexing="ij"
    )
    mx_ravel, my_ravel = mx.ravel(), my.ravel()
    
    _corrected_tp = xr.DataArray(
        mx, 
        dims=['longitude', 'latitude'],
        coords=dict(
            latitude=config.tp_cf.latitude.values,
            longitude=config.tp_cf.longitude.values
        )
    )
    
    time_values = config.tp_ds.time.values
    batch_size = 48
    
    for batch_start in tqdm(range(0, len(time_values), batch_size), desc="Downscaling tp (batches of 96)"):
        batch_end = min(batch_start + batch_size, len(time_values))
        batch_times = time_values[batch_start:batch_end]
        
        batch_results = []
        
        for time_step in batch_times:
            tp_ds_sub_template.array[:] = config.tp_ds.sel(time=time_step).values
            
            corrected_tp = _corrected_tp.copy()
            corrected_tp[:] = tp_ds_sub_template.bivariate(
                coords=dict(longitude=mx_ravel, latitude=my_ravel), 
                num_threads=2
            ).reshape(mx.shape)
            
            corrected_tp = corrected_tp.expand_dims(time=[time_step]).to_dataset(name='tp')
            
            corrected_tp = xr.where(
                corrected_tp > config.tp_drizzle.sel(time=get_month_of_year(time_step)).drop_vars('time'), 
                corrected_tp, 
                0.0
            )
            corrected_tp = corrected_tp * config.tp_cf.sel(time=get_adjusted_day_of_year(time_step)).drop_vars('time')
            corrected_tp = xr.where(config.mask_ds.notnull(), corrected_tp, np.nan)
            
            batch_results.append(corrected_tp)
        
        batch_ds = xr.concat(batch_results, dim='time')
        batch_ds = batch_ds.chunk(config.chunks)
        
        batch_ds.to_zarr(
            config.save_folder / 'tp.zarr', 
            mode='a', 
            region={
                "time": slice(batch_start, batch_end), 
                "latitude": slice(None, None), 
                "longitude": slice(None, None)
            }
        )
start_time = time.time()
config = Config(input_dir="/scratch/depfg/7006713/temp/1km_forcing/input",
				save_folder='/scratch/depfg/7006713/temp/1km_forcing/output',
    			chunks={'time': 1, 'latitude': 21600, 'longitude': 450},
    
				coords={"latitude": slice(14, 1.0), "longitude": slice(8, 17)},
        
				mask_ds='gwStorage.nc',
            
				tas_ds='forcing/W5E5/tas_W5E5v2.0_19790101-20191231.nc',
				tas_cf='forcing/correctionFactors/ta_correction_factor',
				
				evap_ds='forcing/W5E5/refPotEvap_W5E5v2.0_1979_2019.nc',
				evap_cf='forcing/correctionFactors/pet_correction_factor',
				
				tp_ds='forcing/W5E5/pr_W5E5v2.0_19790101-20191231_mm_per_day.nc',
				tp_cf='forcing/correctionFactors/tp_correction_factor',
				tp_drizzle='forcing/correctionFactors/dry_days_cutoff',
				)
end_time = time.time()
print(f"⚙️ Config {end_time - start_time:.2f} seconds")

start_time = time.time()
downscale_tas(config)
end_time = time.time()
print(f"🌡️ Temp {end_time - start_time:.2f} seconds")

start_time = time.time()
corrected_evap = downscale_evap(config)
end_time = time.time()
print(f"💨 Evapotranspiration {end_time - start_time:.2f} seconds")

start_time = time.time()
corrected_tp = downscale_tp(config)
end_time = time.time()
print(f"🌧️ Precip {end_time - start_time:.2f} seconds")


for file in ['tas', 'evap', 'tp']:
	ds = xr.open_zarr(config.save_folder / f'{file}.zarr')
	ds = ds.transpose('latitude', 'longitude', 'time')
	ds = ds.compute()
	ds.to_netcdf(config.save_folder / f'{file}_1990_2019.nc')
