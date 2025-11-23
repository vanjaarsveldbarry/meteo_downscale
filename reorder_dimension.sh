#!/bin/bash

source ${HOME}/.bashrc
mamba activate meteo_ds

# Set input and output folders
input_folder="/scratch/depfg/7006713/temp/1km_forcing/output"
save_folder="/scratch/depfg/7006713/cameroon_forcing"

# Create output folder if it doesn't exist
mkdir -p "${save_folder}"

# List of files to process
files=("evap_1990_2019.nc" "tas_1990_2019.nc" "tp_1990_2019.nc")

# Desired dimension order
dim_order="time,latitude,longitude"

# Process each file in parallel
for file in "${files[@]}"; do
    (
        echo "Processing ${file}..."
        
        input_file="${input_folder}/${file}"
        output_file="${save_folder}/${file}"
        
        # Reorder dimensions (time becomes UNLIMITED automatically)
        ncpdq -a ${dim_order} "${input_file}" "${output_file}"
        
        echo "Completed ${file}"
    )
done

# Wait for all background processes to finish
wait

echo "All files processed successfully!"
