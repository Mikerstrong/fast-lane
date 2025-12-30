# Portainer Stack (copy/paste)

Paste this into **Portainer → Stacks → Add stack → Web editor**.

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
