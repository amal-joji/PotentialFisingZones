# PotentialFisingZones
📌 Overview

This project aims to identify and forecast Potential Fishing Zones (PFZs) along the Tamil Nadu Coast, India, using satellite-derived oceanographic parameters and machine learning techniques.

The system integrates Chlorophyll-a concentration, Sea Surface Temperature (SST), and Ocean Current data to generate PFZ probability maps and forecast the movement of fishing zones for the next 24, 48, and 72 hours.

By leveraging remote sensing and spatio-temporal machine learning, the project provides a data-driven approach to assist fishermen in locating regions with a high probability of fish aggregation.

🎯 Objectives
Predict Potential Fishing Zones using satellite-derived oceanographic parameters.
Generate PFZ probability maps for the Tamil Nadu coast.
Forecast PFZ movement using ocean current advection.
Build a machine learning model for dynamic PFZ prediction.
Visualize PFZ hotspots through geospatial heatmaps.

🛰️ Data Sources
1. NASA MODIS Aqua Ocean Color Dataset
Parameter: Chlorophyll-a Concentration (chlor_a)
Resolution: 4 km
Source: NASA OceanColor
2. NASA MODIS Sea Surface Temperature Dataset
Parameter: Sea Surface Temperature (sst)
Resolution: 4 km
Source: NASA OceanColor
3. Copernicus Marine Environment Monitoring Service (CMEMS)
Parameters:
Zonal Current Velocity (uo)
Meridional Current Velocity (vo)
Source: CMEMS Global Ocean Physics Dataset
