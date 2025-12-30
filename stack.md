# Portainer Stack (copy/paste)

Portainer has two ways to deploy this stack:

## Recommended: Git repository (pulls from GitHub)

Use these values in **Portainer → Stacks → Add stack → Git repository**:

- Repository URL: `https://github.com/Mikerstrong/fast-lane.git`
- Reference: `main`
- Compose path: `docker-compose.yml`

Then click **Deploy the stack**.

## Web editor (YAML only)

Docker Compose YAML cannot “pull from Git” by itself.

Only use the Web editor if Portainer already has the project files available on the host (so it can build using `Dockerfile`).

```yaml
services:
  fast-lane:
    container_name: fast-lane
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8569:8501"
    environment:
      - FASTLANE_DATA_DIR=/data
    volumes:
      - fastlane_data:/data
    restart: unless-stopped

volumes:
  fastlane_data:
```

Important:
- The `build:` section requires Portainer to have access to the repo files (`Dockerfile`, `app.py`, etc.).
- If you are using the Web editor only, Portainer does not automatically upload your source folder.
  - Use **Add stack → Git repository** and set **Compose path** to `docker-compose.yml`, OR
  - Build/push an image first, then replace `build:` with `image:`.
