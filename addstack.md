# Deploy "Fast Lane" in Portainer (Add Stack)

This project runs a Streamlit app on port **8501**.

## Option A — Portainer "Web editor" (paste compose)

1. Open **Portainer**
2. Go to **Stacks** → **Add stack**
3. **Name**: `fast-lane`
4. In **Web editor**, paste this `docker-compose.yml`:

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

5. Portainer needs the project files (Dockerfile + app code). With the Web editor method, Portainer does **not** automatically upload your source folder.

Use **Option B (Git repository)** unless you already have the files available on the Portainer host.

## Option B — Portainer "Git repository" (recommended)

1. Push this repo to GitHub (already set up as `Mikerstrong/fast-lane`).
2. In **Portainer** → **Stacks** → **Add stack**
3. Choose **Repository** / **Git repository**
4. Repo URL:

- `https://github.com/Mikerstrong/fast-lane.git`

5. (Optional) Reference: `main`
6. Compose path: `docker-compose.yml`
7. Deploy the stack.

## After deploy

- Open: `http://<your-server-ip>:8501`

## Persistent data

Your cache + portfolio files are stored in a named volume mounted at `/data`.
The app uses `FASTLANE_DATA_DIR=/data`, so these files persist across restarts:

- `stock_analysis_cache.json`
- `mystocks.json`
- `bulk_debug.log`
