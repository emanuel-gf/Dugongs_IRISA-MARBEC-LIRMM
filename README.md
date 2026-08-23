Exploring latent space for object detection.
This is a compiled of my master's thesis.

What you will find here:

Exploring and clustering high-dimensional vectors.
Developing uniqueness notions based on nearest-neighboors and cosine similarity.
Working with three strategy for efficient data labeling using unsupervised methods.
These three approaches are based on centroid, uniqueness and ball-radius, an attempt to rank a set of embeddings by their uniqueness and remove redudancy by a given radius of a ball drawn from a anchor embedding,
RTDETR model config. Using Omega and yaml for orchestrating deployment.
Extracting embeddings from dinov3.
What you will not going to find:

Dataset. Due to data sensitivity of vulnerable species.
Reminder to myself:
kill the mongodb instance in the server: pkill -u $(whoami) -f mongod

launch a new child process of mongod to be able of connecting through notebook:
mongod --dbpath /share/home/USER/fiftyone/mongodb --port 44123 --fork --logpath /share/home/USER/fiftyone/mongodb/log/mongo.log
