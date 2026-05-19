{{ config(
    materialized='external',
    location=var('gold_dir') ~ '/fact_audio_features.parquet',
    format='parquet'
) }}

with audio as (
    select * from {{ ref('stg_silver_audio_features') }}
),
track_keys as (
    select distinct
        track_id,
        primary_artist_id,
        album_id
    from {{ ref('stg_silver_tracks') }}
)

select
    a.track_id,
    t.primary_artist_id,
    t.album_id,
    a.danceability,
    a.energy,
    a.valence,
    a.tempo,
    a.key,
    a.loudness,
    a.time_signature,
    a.speechiness,
    a.acousticness,
    a.instrumentalness,
    a.liveness,
    a.mode,
    a.duration_ms
from audio a
left join track_keys t on a.track_id = t.track_id
