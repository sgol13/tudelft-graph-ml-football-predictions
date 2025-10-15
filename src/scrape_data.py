import pickle

import soccerdata as sd
from tqdm import tqdm


def scrape_data(season: int):
    ws = sd.WhoScored(leagues="ENG-Premier League", seasons=season)

    epl_schedule = ws.read_schedule()

    matches = epl_schedule.to_dict(orient="records")

    for match in tqdm(matches):
        game_id = match["game_id"]
        events = ws.read_events(match_id=game_id)
        match["events"] = events

    filename = f"data/epl_{season}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(matches, f)


def main():
    for year in range(2015, 2024):
        scrape_data(year)


if __name__ == "__main__":
    main()
