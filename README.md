# Exoplanet_Detection-Habitability
24/25 School year research project on the detection of exoplanets using Convolutional Neural Networks and the evaluation of their habitability

Requires user to manually download Data from the NASA Exoplanet Archive
Process:
  1. Download dataset from NASA Exoplanet Archive Bulk Data Download HERE (data I used was kepler_confirmed_wget.bat) ----> https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/
  2. Run downloaded wget script using step by step process available HERE ----> http://irsa.ipac.caltech.edu/docs/batch_download_help.html
  3. Edit LightCurveDataset code variable DATA_DIRECTORY with directory of bulk downloaded data
  4. Run

Abstract: Exoplanet discovery has been a significant hotspot in recent years, with the boundaries of modern technology being pushed in order to discover habitable worlds beyond our solar system. This project focuses on the classification of different Kepler Objects of Interest (KOI), using a one dimensional convolutional neural network (1-D CNN)  along with several optimizations to the network to raise accuracy. In order to train the model, data taken from the NASA Exoplanet Archive-roughly 2,500 confirmed planets and an equal amount of generated false positives-needed to be preprocessed. Processing included taking the Pre-search Data Conditioning Simple Aperture Photometry (PDCSAP) flux, normalizing the conditioned flux, trimming NaN values, centering the exoplanet transit dip, and truncating or padding the data to a fixed time interval. I was able to train my deep learning neural network using optimization algorithms to achieve an accuracy of 95.43%. Finally, I created a robust random forest classifier capable of predicting the habitability of a given exoplanet given data parameters from the NASA Exoplanet Archive, which had an average prediction accuracy of 99%. This deep learning pipeline could help further the field of exoplanet discovery by increasing efficiency and potentially help discover a real habitable planet.
