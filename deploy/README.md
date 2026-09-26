# Deployment

One Linux server with Docker runs the whole public site:

```
https://DOMAIN  ->  Caddy (HTTPS, static website from SITE_DIR)
                    └── /api/*  ->  demo container (FastAPI, demo/app.py, CPU inference)
```

## One-time server setup
1. Point the domain's DNS `A` record at the server's IP.
2. Install Docker, create a deploy user that may run it, and add the deploy SSH public key to
   `~deploy/.ssh/authorized_keys`.
3. `sudo apt install rsync && sudo mkdir -p /srv/site /opt/traffic && sudo chown deploy /srv/site /opt/traffic`

## GitHub configuration (both repositories)
Secrets (Settings → Secrets and variables → Actions → Secrets):

| Secret | Value |
|---|---|
| `SSH_HOST` | server IP or hostname |
| `SSH_USER` | deploy user, e.g. `deploy` |
| `SSH_PRIVATE_KEY` | private key matching the server's authorized key |
| `SSH_PORT` | usually `22` |

Variables (same page → Variables):

| Variable | Value |
|---|---|
| `DEPLOY_ENABLED` | `true` once the server exists; deploy jobs are skipped until then |
| `DOMAIN` | e.g. `traffic.example.uz` |
| `DEPLOY_DIR` | `/opt/traffic` (compose files, this repo only) |
| `SITE_DIR` | `/srv/site` (website build) |

The demo image is published to GHCR as `ghcr.io/ctrl-alt-elite-newuu/traffic-event-detection-demo`.
Make the package public (Package settings → Change visibility) so the server can pull it without a token.

The website repository (`traffic-event-detection-web`) needs the same four secrets and the
`DEPLOY_ENABLED` and `SITE_DIR` variables; `API_BASE` is optional (defaults to the same origin).

## Branch flow
All work is pushed to `dev`; `main` only changes through a pull request from `dev`.

| Event | CI checks | Demo image | Deploy to server |
|---|---|---|---|
| push to `dev` | run | built, not published | skipped |
| pull request into `main` | run | built, not published | skipped |
| merge into `main` | run | built and published to GHCR | runs (when `DEPLOY_ENABLED` is `true`) |
