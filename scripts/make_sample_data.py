from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_sample_data(seed: int = 42, n_tracks: int = 220) -> None:
    rng = np.random.default_rng(seed)

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    start_date = pd.Timestamp("2021-01-01")
    regions = ["Global"]
    rows = []
    audio_rows = []

    for i in range(n_tracks):
        track_id = f"{i:022d}"[-22:]
        artist_id = rng.integers(0, 55)
        artist = f"Artist {artist_id}"
        track = f"Track {i}"

        first_date = start_date + pd.Timedelta(days=int(rng.integers(0, 500)))
        base_rank = int(np.clip(rng.normal(115, 45), 1, 200))
        quality = rng.normal(0, 1)
        trend = rng.normal(0, 8) - quality * 10

        base_streams = max(5000, rng.normal(250_000, 90_000) + (200 - base_rank) * 2500 + quality * 60_000)

        for region in regions:
            for d in range(45):
                rank = int(np.clip(base_rank + trend * d / 7 + rng.normal(0, 9), 1, 200))
                # Missing days imitate chart gaps/disappearances.
                if d > 12 and rank > 185 and rng.random() < 0.45:
                    continue

                streams = max(1000, base_streams * (1 + 0.04 * quality) ** d + rng.normal(0, 25_000))
                rows.append(
                    {
                        "date": first_date + pd.Timedelta(days=d),
                        "rank": rank,
                        "title": track,
                        "artist": artist,
                        "region": region,
                        "chart": "top200",
                        "trend": "SAME_POSITION",
                        "streams": int(streams),
                        "url": f"https://open.spotify.com/track/{track_id}",
                    }
                )

        energy = float(np.clip(rng.normal(0.58 + 0.06 * quality, 0.18), 0, 1))
        danceability = float(np.clip(rng.normal(0.62 + 0.04 * quality, 0.15), 0, 1))
        valence = float(np.clip(rng.normal(0.50, 0.20), 0, 1))
        tempo = float(np.clip(rng.normal(120 + 6 * quality, 25), 60, 210))
        loudness = float(np.clip(rng.normal(-7 + 1.5 * quality, 3), -35, 0))

        audio_rows.append(
            {
                "track_id": track_id,
                "track_name": track,
                "artists": artist,
                "popularity": int(np.clip(45 + 12 * quality + rng.normal(0, 8), 0, 100)),
                "duration_ms": int(np.clip(rng.normal(200_000, 35_000), 95_000, 420_000)),
                "explicit": bool(rng.random() < 0.18),
                "danceability": danceability,
                "energy": energy,
                "key": int(rng.integers(0, 12)),
                "loudness": loudness,
                "mode": int(rng.integers(0, 2)),
                "speechiness": float(np.clip(rng.beta(2, 10), 0, 1)),
                "acousticness": float(np.clip(rng.beta(2, 5), 0, 1)),
                "instrumentalness": float(np.clip(rng.beta(1, 20), 0, 1)),
                "liveness": float(np.clip(rng.beta(2, 8), 0, 1)),
                "valence": valence,
                "tempo": tempo,
                "time_signature": 4,
                "track_genre": rng.choice(["pop", "rock", "hip-hop", "dance", "indie"]),
            }
        )

    charts = pd.DataFrame(rows).sort_values(["date", "rank"])
    audio = pd.DataFrame(audio_rows)

    charts.to_csv(raw_dir / "charts_sample.csv", index=False)
    audio.to_csv(raw_dir / "tracks_sample.csv", index=False)

    print(f"Saved {len(charts):,} chart rows to {raw_dir / 'charts_sample.csv'}")
    print(f"Saved {len(audio):,} audio rows to {raw_dir / 'tracks_sample.csv'}")


if __name__ == "__main__":
    make_sample_data()
