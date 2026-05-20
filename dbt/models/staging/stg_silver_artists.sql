{{ config(materialized='view') }}

select
    artist_id,
    name                                as artist_name,
    cast(popularity as integer)         as artist_popularity,
    followers,
    genres,
    primary_genre
from read_parquet('{{ var("silver_dir") }}/artists.parquet')
