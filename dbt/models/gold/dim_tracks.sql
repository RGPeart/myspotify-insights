{% if target.type == 'duckdb' %}
{{ config(
    materialized='external',
    location=var('gold_dir') ~ '/dim_tracks.parquet',
    format='parquet'
) }}
{% else %}
{{ config(materialized='table') }}
{% endif %}

with tracks as (
    select * from {{ ref('stg_silver_tracks') }}
),
artists as (
    select artist_id, primary_genre, artist_popularity
    from {{ ref('stg_silver_artists') }}
),
joined as (
    select
        t.*,
        a.primary_genre        as artist_primary_genre,
        a.artist_popularity    as artist_popularity
    from tracks t
    left join artists a on t.primary_artist_id = a.artist_id
),
median_ap as (
    select coalesce(
        median(artist_popularity),
        0
    ) as value
    from joined
    where artist_popularity is not null
)

select
    j.track_id,
    j.name,
    j.track_popularity,
    j.duration_ms,
    j.explicit,
    j.primary_artist_id,
    j.primary_artist_name,
    j.album_id,
    j.album_name,
    j.release_date,
    coalesce(j.artist_primary_genre, 'unknown') as primary_genre,
    least(
        1.0,
        greatest(
            0.0,
            0.6 * coalesce(j.track_popularity, 0) / 100.0
            + 0.4 * coalesce(j.artist_popularity, m.value) / 100.0
        )
    ) as composite_popularity
from joined j
cross join median_ap m
