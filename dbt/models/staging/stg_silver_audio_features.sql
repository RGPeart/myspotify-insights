{{ config(materialized='view') }}

select
    track_id,
    danceability,
    energy,
    valence,
    tempo,
    key,
    loudness,
    time_signature,
    speechiness,
    acousticness,
    instrumentalness,
    liveness,
    mode,
    duration_ms
from read_parquet('{{ var("silver_dir") }}/audio_features.parquet')
