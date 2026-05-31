# Remote sensing for landcover detection: ML techniques

This repository contain course material for the MLESS lecture by Prof. Dr. Martin Schultz at the University of Cologne, Germany.
Two Jupyter notebooks demonstrate the use of random forests and convolutional neural networks for a simplified landcover classification task
based on satellite remote sensing data.

The data for these demonstrations is a subset from the [SAT-6](https://csc.lsu.edu/~saikat/deepsat/) dataset by
_Saikat Basu, Sangram Ganguly, Supratik Mukhopadhyay, Robert Dibiano, Manohar Karki and Ramakrishna Nemani, DeepSat - A Learning framework for Satellite Imagery, ACM SIGSPATIAL 2015._
It consists of 28x28 pixels uint8 images with 4 channels and labels of 6 landcover classes - barren land, trees, grassland, roads, buildings and water bodies.

The classification task is scene classification, i.e. the entire 28x28 pixel image is classified a sone landcover type.

## Download the data

Unfortunately, there is no anonymous and free access to the Deepsat data. However, a subset has been extracted and made available atthe B2SHARE server at FZ Jülich: 
https://b2share.eudat.eu/records/89654eac10724d30a6c7e51f2c5422de. Thi scomprises only the test set of the original data - for our educational experiments, this is sufficient.

The three data files must be stored in a `data` directory in the same path as the notebook itself.

## Run the example notebooks

Start with the Random_forest_classifier notebook. WARNING: loading the data into pandas consumes ~5 GBytes of memory. Make sure that your Jupyter lab has sufficient memory.

Once you fully understood what this notebook does, take a look at the CNN_classifier notebook and run it. Note that you need to have Pytorch installed to run the CNN_classifier notebook.

Compare training times, inference times (if you notice a difference) and the quality of the results.

Think about the network and training paraneters: which ones would you modify if you want to improve the results?


## Author

Martin Schultz, April 2026

