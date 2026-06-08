"""One-time interactive login to obtain a Spotify refresh token.

Usage::

    python -m src.ingestion.spotify_auth_login

This opens a browser, asks the user to grant the scopes declared in
``spotify_auth.SCOPES``, then prints the resulting refresh token. Copy that
value into ``.env`` as ``SPOTIFY_REFRESH_TOKEN`` so headless runs (Airflow,
CI) can use it.

The script writes the token to stdout only; nothing is persisted to disk.
"""

import sys

from dotenv import load_dotenv
from spotipy.cache_handler import MemoryCacheHandler

from src.ingestion.spotify_auth import SCOPES, build_oauth


def main() -> int:
    load_dotenv()
    cache_handler = MemoryCacheHandler()
    oauth = build_oauth(open_browser=True, cache_handler=cache_handler)

    # Triggers the browser flow and writes the resulting token_info into the
    # in-memory cache handler.
    token_info = oauth.get_access_token(as_dict=True)
    refresh_token = token_info.get("refresh_token") if token_info else None
    if not refresh_token:
        print("ERROR: did not receive a refresh token from Spotify.", file=sys.stderr)
        return 1

    print()
    print("=" * 72)
    print("Spotify refresh token obtained.")
    print("Scopes granted:", " ".join(SCOPES))
    print()
    print("Copy the line below into your .env file:")
    print()
    print(f"SPOTIFY_REFRESH_TOKEN={refresh_token}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
