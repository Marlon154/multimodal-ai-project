# multimodal-ai-project
Project for Multimodal AI class at TUD

# Preare data
- make sure that all processed images are in `/data/nytimes/images_processed`
- prepare the mongodb using the docker compose (check the `docker-compose.yml`)

# Docker
```
# build training container
docker build -t training . 
# run compose
docker compose up -d
```

# Links
[Overleaf](https://sharelatex.tu-darmstadt.de/project/6654a0cdac6c54d019b61b3a)

# Authors
- Jonas Milkovits
- Marlon May
