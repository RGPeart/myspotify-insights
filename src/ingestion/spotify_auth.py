"""Spotify user-OAuth helpers.

The Airflow workers run headless, so interactive browser-based auth is not viable
at run time. Instead, a refresh token is obtained once via
``python -m src.ingestion.spotify_auth_login`` and stored in ``.env`` as
``SPOTIFY_REFRESH_TOKEN``. At ingest time, :func:`get_authenticated_client`
seeds spotipy's in-memory token cache with that refresh token and lets spotipy
exchange it for short-lived access tokens automatically.
"""

import os

import spotipy
from spotipy.cache_handler import MemoryCacheHandler
from spotipy.oauth2 import SpotifyOAuth

SCOPES = [
    "user-top-read",
    "user-follow-read",
    "user-read-private",
]

_MISSING_TOKEN_MSG = (
    "SPOTIFY_REFRESH_TOKEN is not set. Run "
    "`python -m src.ingestion.spotify_auth_login` once locally and copy the "
    "printed refresh token into your .env file."
)


def build_oauth(open_browser: bool = False, cache_handler=None) -> SpotifyOAuth:
    """Construct a ``SpotifyOAuth`` configured with the project's scopes."""
    return SpotifyOAuth(
        client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
        client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=os.environ.get("SPOTIFY_REDIRECT_URI"),
        scope=" ".join(SCOPES),
        open_browser=open_browser,
        cache_handler=cache_handler,
    )


def get_authenticated_client() -> spotipy.Spotify:
    """Return a ``spotipy.Spotify`` instance that auto-refreshes from a stored refresh token.

    Reads ``SPOTIFY_REFRESH_TOKEN`` from the environment and primes an in-memory
    token cache so the first API call triggers an access-token refresh without
    requiring any browser interaction.
    """
    refresh_token = os.environ.get("SPOTIFY_REFRESH_TOKEN")
    if not refresh_token:
        raise RuntimeError(_MISSING_TOKEN_MSG)

    token_info = {
        "access_token": "",
        "refresh_token": refresh_token,
        "expires_at": 0,
        "scope": " ".join(SCOPES),
        "token_type": "Bearer",
    }
    cache_handler = MemoryCacheHandler(token_info=token_info)
    oauth = build_oauth(open_browser=False, cache_handler=cache_handler)
    client = spotipy.Spotify(auth_manager=oauth)

    # Force an immediate refresh + auth check so a revoked/invalid token surfaces
    # here with an actionable message, rather than as an opaque SpotifyException
    # deep inside the first ingest call.
    try:
        client.current_user()
    except spotipy.SpotifyException as exc:
        raise RuntimeError(
            "SPOTIFY_REFRESH_TOKEN was rejected by Spotify "
            f"(status={exc.http_status}). The token may have been revoked or the "
            "granted scopes changed. Re-run `python -m src.ingestion.spotify_auth_login` "
            "and update your .env."
        ) from exc

    return client
