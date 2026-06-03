## TO RUN ON THE CLOUD 

## preprocessed the data before starting the slide
## import embeddings <s
import os

# Define the URI to point to your manual process
os.environ["FIFTYONE_DATABASE_URI"] = "mongodb://localhost:44123"

import fiftyone as fo

# Verify connection
print(fo.core.odm.database.get_db_conn()) 

dataset = fo.load_dataset("FLPLAN")

session = fo.launch_app(dataset,
                        port=5151,
                        auto=False)
print(session.url)  
session.wait()