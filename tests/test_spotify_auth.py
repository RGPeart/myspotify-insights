import pytest
from unittest.mock import patch

from src.ingestion import spotify_auth


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "test_id")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "test_secret")
    monkeypatch.setenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")


class TestGetAuthenticatedClient:
    # Without a refresh token we cannot construct an auth manager, so we must fail loudly
    # with an actionable message rather than silently producing a broken client.
    def test_raises_when_refresh_token_missing(self, env, monkeypatch):
        monkeypatch.delenv("SPOTIFY_REFRESH_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="SPOTIFY_REFRESH_TOKEN"):
            spotify_auth.get_authenticated_client()

    # The in-memory token cache must be seeded with the refresh token so spotipy can
    # exchange it for an access token without any disk I/O or browser interaction.
    def test_seeds_cache_with_refresh_token(self, env, monkeypatch):
        monkeypatch.setenv("SPOTIFY_REFRESH_TOKEN", "rt-123")
        with patch("src.ingestion.spotify_auth.SpotifyOAuth") as mock_oauth, \
             patch("src.ingestion.spotify_auth.spotipy.Spotify") as mock_sp:
            spotify_auth.get_authenticated_client()

        cache_handler = mock_oauth.call_args.kwargs["cache_handler"]
        assert cache_handler.get_cached_token()["refresh_token"] == "rt-123"
        # Auth manager is wired into the returned Spotify client.
        assert mock_sp.call_args.kwargs["auth_manager"] is mock_oauth.return_value

    # All three project scopes must be requested or the user-data endpoints will 403.
    def test_requests_all_required_scopes(self, env, monkeypatch):
        monkeypatch.setenv("SPOTIFY_REFRESH_TOKEN", "rt-123")
        with patch("src.ingestion.spotify_auth.SpotifyOAuth") as mock_oauth, \
             patch("src.ingestion.spotify_auth.spotipy.Spotify"):
            spotify_auth.get_authenticated_client()

        scope_str = mock_oauth.call_args.kwargs["scope"]
        for s in ("user-top-read", "user-follow-read", "user-read-private"):
            assert s in scope_str
