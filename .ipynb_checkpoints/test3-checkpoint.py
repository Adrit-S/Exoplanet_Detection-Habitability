from astropy.table import Table

# Load the TBL file
file_path = "D:\Exoplanet Dataset\kplr007207061-2013098041711_slc_lc.tbl"
tbl_data = Table.read(file_path, format="ascii")

# Print all available column names
print(tbl_data.colnames)
