# Multimodal AI Project
Project for Multimodal AI class at TUD by Prof. Anna Rohrbach and Prof. Marcus Rohrbach. In this project we approached the Transform and Tell model, an image captioning model for news articles, investigated possible improvements for the existing architecture, and analyzed the 800k dataset to find new ways in which we can enhance the modalities used in the process of captioning images for news articles.

## Data
### Setup for Sample Testing
> [MongoDB Sample](https://drive.google.com/file/d/1yCZ0Qp21sDa7fnZvI8mnvKvD83UjqaIq/view?usp=sharing)
> [Image Sample]()
> **IMPORTANT**: change `self.client.nytimes` to `self.client.nytimes_sample` in the source code when using the sample file 
```
# Download mongo db sample
# move it to the ./data/dump folder
cd .../multimodal-ai-project/data
mv ~/Downloads/nytimes-2020-04-21_sample.gz ./dump

# Download image sample
# Check the official readme on where to put it and extract it

# run the docker container
# make sure that you are in ./data/
docker compose up -f docker-compose-restore.yml

# check out the web interface on localhost:8081
```

### Files
- `mongo_db/columns.txt`: contains an overview of all columns of all collections of the complete mongo db
- `docker-compose.yml`: used to launch multiple docker containers that start the db, a gui and restore a potential backup
- `dump/nytimes-2020-04-21_sample.gz`: sample dataset used to debug any architecture and training code

### Scene Embedding
If you want to use the scene embedding, run the following:
```
docker compose up -f docker-compose-scene-embedder.yml -d --build
```

### Docker Specifics
- Restoring the database currently saves all restored files on a volume (`./export`) and makes the DB persistent
- this can be changed by removing the volume from `mongo_db` in `docker-compose.yml`
- if this is removed, the `mongo_restore` has to be performed on every startup
- if it is not removed, the `mongo_restore` can be commented out/removed after the first startup

### Dump existing database
> make sure to adjust database, file and network names
```
docker run -it --rm -v ./dump:/dump --network data_mongo_net mongo bash
mongodump --host data-mongo_db-1 --port 27017 --username root --password secure_pw --authenticationDatabase admin --db nytimes --gzip --archive=/dump/nytimes-2020-04-21_sample.gz
```

### MongoDB
- put the dump in `data/dump/`
- contains three collections: 'articles', 'images', 'objects'
- all columns can be found in `columns.txt`
- 'foreign key'-linkage:
    - `objects._id -> images._id & images.captions.id -> articles._id`
- **IMPORTANT**: change `self.client.nytimes` to `self.client.nytimes_sample` when using the sample file 
```
docker compose up -d # start docker containers
# access gui at localhost:8081
```

## Training 
To run the project with docker, the following commands can be used:
```
docker compose up -d --build
```
Note to track the runs to W&B a API Key is needed in the environment with the variable name `WANDB_API_KEY`.
All results will be put into the `output_dir` configured in `config.yml`.

### Contexts
To adjust the contexts, make sure to adjust the `config.yml`.
Following modalities are avaiable:
- 'image': the image to caption
- 'article': the context extracted from the article (can be further configured by 'context_before/after')
- 'faces': embeddings of faces of persons in the image (already in DB Dump)
- 'objects': embeddings of objects in the image (already in DB Dump)
- 'scene_embeddings': generated scene embeddings for the image

## Authors
- [Jonas Milkovits](https://github.com/j-milkovits)
- [Marlon May](https://github.com/Marlon154)
- [Laurenz Kammeyer](https://github.com/DeadCowboy)
- [Frederick Wichert](https://github.com/f-wichert)
