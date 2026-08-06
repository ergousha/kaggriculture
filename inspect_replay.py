import glob
import json

replays = glob.glob("replays/*.json")
with open(replays[0]) as f:
    data = json.load(f)

step = data["steps"][100]  # Check a bit later
farm = step[0]["observation"]["farms"][0]
print("Tiles count:", len(farm.get("tiles", [])))
if len(farm.get("tiles", [])) > 0:
    print("Example tile:", farm["tiles"][0])
