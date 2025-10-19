import pandas as pd
import random
from datetime import datetime, timedelta

crime_types = ["robbery", "homicide"]
suspects = ["John Doe", "Alice Green", "Marcus Lee", "Sara Patel", "Evan Cole"]
victims = ["Tom Smith", "Linda Brown", "Robert King", "Nina Park", "David Ross"]
objects = ["knife", "handgun", "wallet", "phone", "crowbar", "jewelry"]
locations = ["downtown street", "convenience store", "apartment", "alleyway", "parking lot"]
actions_robbery = ["stole", "threatened", "fled", "entered"]
actions_homicide = ["attacked", "argued with", "stabbed", "shot"]

records = []
scene_id = 1

for _ in range(100):  # Generate 100 crime scenes
    crime_type = random.choice(crime_types)
    scene_id += 1
    base_time = datetime(2024, 5, 1, random.randint(0, 23), random.randint(0, 59))
    suspect = random.choice(suspects)
    victim = random.choice(victims)
    location = random.choice(locations)
    
    if crime_type == "robbery":
        events = [
            (suspect, victim, random.choice(objects), location, random.choice(actions_robbery), base_time),
            (suspect, victim, random.choice(objects), location, "fled", base_time + timedelta(minutes=5))
        ]
    else:
        events = [
            (suspect, victim, random.choice(objects), location, "argued with", base_time),
            (suspect, victim, random.choice(objects), location, random.choice(actions_homicide), base_time + timedelta(minutes=10))
        ]

    for event_id, (s, v, o, l, a, t) in enumerate(events, 1):
        notes = f"{s} {a} {v} using a {o} at {l} around {t.strftime('%H:%M')}."
        records.append({
            "scene_id": scene_id,
            "event_id": event_id,
            "crime_type": crime_type,
            "suspect": s,
            "victim": v,
            "object": o,
            "location": l,
            "action": a,
            "timestamp": t.isoformat(),
            "notes": notes
        })

df = pd.DataFrame(records)
df.to_csv("data/synthetic_crime_scenes.csv", index=False)
print(df.head())
