# Multimodal AI Project
Project for Multimodal AI class at TUD by Prof. Anna Rohrbach and Prof. Marcus Rohrbach.

# Preare data
- make sure that all processed images are in `/data/nytimes/images_processed`

Further, ensure that the monodb dump is in the folder /dump/nytimes-2020-04-21 or update the path in the docker-compose-restore.yml file.
To restore the database, the following command can be used:
```bash
docker compose --file docker-compose-restore.yml up -d --build
```

# Docker
To run the project with docker, the following commands can be used:
```bash
docker compose up -d --build
```
Note to track the runs to W&B a API Key is needed in the environment with the variable name `WANDB_API_KEY`.
For the database follow the [instructions from Transform and Tell](https://github.com/alasdairtran/transform-and-tell?tab=readme-ov-file#getting-data).

# Links
[Overleaf](https://sharelatex.tu-darmstadt.de/project/6654a0cdac6c54d019b61b3a)

# Authors
- [Jonas Milkovits](https://github.com/j-milkovits)
- [Marlon May](https://github.com/Marlon154)
- [Laurzen Kammeyer](https://github.com/DeadCowboy)
- [Frederick Wichert](https://github.com/f-wichert)
