import os
import math
import logging
from collections import Counter, defaultdict
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from supabase import create_client, Client
from dotenv import load_dotenv
import uvicorn

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase configuration. Please check .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Constants
# ---------------------------
WEIGHTS = {
    "activity_type": 2,
    "tag_match": 1,
    "nearby_bonus": 1
}

# ---------------------------
# Utility Functions
# ---------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_user_preferences(user_id: str):
    reg_response = supabase.table("registrations").select("event_id").eq("volunteer_id", user_id).execute()
    registration_data = reg_response.data or []

    if not registration_data:
        return {"tags": set(), "activity_types": set()}

    event_ids = [r["event_id"] for r in registration_data]

    # Batch fetch all tags & event types
    tag_response = supabase.table("event_tags").select("event_id, tag_name").in_("event_id", event_ids).execute()
    event_response = supabase.table("events").select("event_id, activity_type").in_("event_id", event_ids).execute()

    tag_counter = Counter()
    act_type_counter = Counter()

    for tag in tag_response.data or []:
        raw = tag["tag_name"]
        individual = [t.strip() for t in raw.split(",") if t.strip()]
        tag_counter.update(individual)

    for event in event_response.data or []:
        act_type_counter[event["activity_type"]] += 1

    return {
        "tags": set([t for t, _ in tag_counter.most_common(5)]),
        "activity_types": set([a for a, _ in act_type_counter.most_common(3)])
    }


def get_nearby_events(user_lat, user_lon, max_distance_km=10):
    response = supabase.table("events").select("event_id, event_name, location_cords, activity_type").execute()
    nearby = []

    for event in response.data or []:
        cords = event.get("location_cords")
        if cords and isinstance(cords, dict):
            try:
                lat, lon = float(cords["latitude"]), float(cords["longitude"])
                dist = haversine(user_lat, user_lon, lat, lon)
                if dist <= max_distance_km:
                    event["distance"] = dist
                    nearby.append(event)
            except Exception as e:
                logger.warning(f"Skipping invalid location: {cords} - {e}")

    return nearby


def score_events(nearby_events, user_preferences):
    event_ids = [e["event_id"] for e in nearby_events]
    tag_data = supabase.table("event_tags").select("event_id, tag_name").in_("event_id", event_ids).execute().data or []

    event_tags_map = defaultdict(set)
    for tag in tag_data:
        tags = [t.strip() for t in tag["tag_name"].split(",") if t.strip()]
        event_tags_map[tag["event_id"]].update(tags)

    for event in nearby_events:
        score = 0

        if event.get("activity_type") in user_preferences["activity_types"]:
            score += WEIGHTS["activity_type"]

        matching_tags = event_tags_map[event["event_id"]].intersection(user_preferences["tags"])
        score += WEIGHTS["tag_match"] * len(matching_tags)

        if event.get("distance") and event["distance"] <= 5:
            score += WEIGHTS["nearby_bonus"]

        event["score"] = score

    return sorted(nearby_events, key=lambda e: (-e["score"], e["distance"]))


def recommend_events(user_id: str, user_lat: float, user_lon: float):
    prefs = get_user_preferences(user_id)
    nearby = get_nearby_events(user_lat, user_lon)

    if not nearby:
        logger.info("No nearby events found. Expanding search radius to 25km.")
        nearby = get_nearby_events(user_lat, user_lon, max_distance_km=25)

    scored = score_events(nearby, prefs)

    # Get full event details based on top N scored event IDs
    top_event_ids = [e["event_id"] for e in scored]

    full_event_response = supabase.table("events").select("*").in_("event_id", top_event_ids).execute()
    full_event_map = {event["event_id"]: event for event in full_event_response.data or []}

    enriched_events = []
    for event in scored:
        full_details = full_event_map.get(event["event_id"], {})
        enriched_event = {
            **full_details,
            "score": event["score"],
            "distance": event["distance"]
        }
        enriched_events.append(enriched_event)

    return enriched_events

# ---------------------------
# Pydantic Models
# ---------------------------
class Event(BaseModel):
    event_id: int
    score: int
    distance: float

    class Config:
        extra = "allow"


class RecommendationsResponse(BaseModel):
    user_id: str
    recommendations: List[Event]

# ---------------------------
# FastAPI App Setup
# ---------------------------
app = FastAPI(
    title="Volunteer Event Recommendation API",
    description="Recommends events based on user preferences and location.",
    version="2.0"
)

@app.get("/recommendations", response_model=RecommendationsResponse)
def get_recommendations(
    user_id: str = Query(..., description="User ID for whom to fetch recommendations"),
    user_lat: float = Query(..., description="User's current latitude"),
    user_lon: float = Query(..., description="User's current longitude")
):
    try:
        recommendations = recommend_events(user_id, user_lat, user_lon)
    except Exception as e:
        logger.error(f"Error retrieving recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return RecommendationsResponse(
        user_id=user_id,
        recommendations=[
            Event(**e) for e in recommendations
        ]
    )

# ---------------------------
# Run the API
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
