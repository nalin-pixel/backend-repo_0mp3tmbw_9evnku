import os
from datetime import datetime, timezone
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr

from database import db, create_document, get_documents

app = FastAPI(title="PriceFix API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------
# Auth Models & Helpers
# ----------------------
class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = None
    password: str
    confirm_password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class GoogleAuthRequest(BaseModel):
    email: EmailStr
    name: Optional[str] = None
    avatar_url: Optional[str] = None


class AuthResponse(BaseModel):
    message: str
    email: EmailStr
    name: Optional[str] = None
    provider: str


# simple password hashing using passlib if available, fallback to sha256
_hasher = None
try:
    from passlib.context import CryptContext

    _pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def hash_password(p: str) -> str:
        return _pwd_context.hash(p)

    def verify_password(p: str, hashed: str) -> bool:
        return _pwd_context.verify(p, hashed)

    _hasher = "passlib"
except Exception:
    import hashlib

    def hash_password(p: str) -> str:
        return hashlib.sha256(p.encode()).hexdigest()

    def verify_password(p: str, hashed: str) -> bool:
        return hash_password(p) == hashed

    _hasher = "sha256"


# ----------------------
# Ride Price Models
# ----------------------
class Coordinate(BaseModel):
    lat: float
    lng: float


class PriceRequest(BaseModel):
    origin: Coordinate = Field(..., description="Origin coordinates")
    destination: Coordinate = Field(..., description="Destination coordinates")
    user_email: Optional[EmailStr] = None


class ProviderQuote(BaseModel):
    provider: str
    price: float
    currency: str = "INR"
    eta_min: int
    notes: Optional[str] = None


class PriceResponse(BaseModel):
    distance_km: float
    quotes: List[ProviderQuote]
    cheapest: ProviderQuote


# ----------------------
# Utility functions
# ----------------------
from math import radians, sin, cos, asin, sqrt


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return round(R * c, 2)


def simulate_quotes(distance_km: float) -> List[ProviderQuote]:
    """Simulate pricing logic for Uber, Ola, Rapido based on distance.
    This is a demo approximation – for production, integrate official APIs.
    """
    # Base and per-km rates (hypothetical)
    pricing = {
        "Uber": {"base": 45, "per_km": 12.5},
        "Ola": {"base": 40, "per_km": 13.0},
        "Rapido": {"base": 30, "per_km": 10.5},
    }
    # Simple surge by distance bands
    surge = 1.0
    if distance_km > 15:
        surge = 1.25
    elif distance_km > 7:
        surge = 1.1

    quotes: List[ProviderQuote] = []
    for provider, cfg in pricing.items():
        price = (cfg["base"] + cfg["per_km"] * max(distance_km, 1)) * surge
        # Add service specific tweaks
        if provider == "Rapido":
            # Cheaper for short rides, not available > 25km
            if distance_km > 25:
                notes = "Not typically available for long distances"
                price = price * 1.5
            else:
                price = price * 0.95
            eta = 6
        elif provider == "Ola":
            eta = 7
        else:  # Uber
            eta = 5
        quotes.append(ProviderQuote(provider=provider, price=round(price, 2), eta_min=eta))
    # determine cheapest
    cheapest = min(quotes, key=lambda q: q.price)
    return quotes, cheapest


# ----------------------
# Routes
# ----------------------
@app.get("/")
def root():
    return {"name": "PriceFix API", "status": "ok"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    return response


@app.post("/api/auth/signup", response_model=AuthResponse)
def signup(payload: SignupRequest):
    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    # check existing
    existing = list(db["user"].find({"email": payload.email})) if db else []
    if existing:
        raise HTTPException(status_code=409, detail="User already exists")

    hashed = hash_password(payload.password)
    user_doc = {
        "name": payload.name,
        "email": payload.email,
        "phone": payload.phone,
        "password_hash": hashed,
        "auth_provider": "password",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    db["user"].insert_one(user_doc)
    return AuthResponse(message="Signup successful", email=payload.email, name=payload.name, provider="password")


@app.post("/api/auth/login", response_model=AuthResponse)
def login(payload: LoginRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    user = db["user"].find_one({"email": payload.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return AuthResponse(message="Login successful", email=user["email"], name=user.get("name"), provider=user.get("auth_provider", "password"))


@app.post("/api/auth/google", response_model=AuthResponse)
def google_auth(payload: GoogleAuthRequest):
    """
    Demo Google auth: trusts the provided email. In production, verify the id_token.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Database not available")
    user = db["user"].find_one({"email": payload.email})
    if not user:
        user = {
            "name": payload.name or payload.email.split("@")[0],
            "email": payload.email,
            "password_hash": "",
            "auth_provider": "google",
            "avatar_url": payload.avatar_url,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        db["user"].insert_one(user)
    return AuthResponse(message="Google sign-in successful", email=user["email"], name=user.get("name"), provider="google")


@app.post("/api/price/estimate", response_model=PriceResponse)
def estimate_prices(payload: PriceRequest):
    # compute distance
    d_km = haversine_km(payload.origin.lat, payload.origin.lng, payload.destination.lat, payload.destination.lng)
    quotes, cheapest = simulate_quotes(d_km)

    # persist query
    try:
        from schemas import RideQuery

        rq = RideQuery(
            user_email=payload.user_email,
            origin=f"{payload.origin.lat},{payload.origin.lng}",
            destination=f"{payload.destination.lat},{payload.destination.lng}",
            distance_km=d_km,
            cheapest_provider=cheapest.provider,
            cheapest_price=cheapest.price,
        )
        create_document("ridequery", rq)
    except Exception:
        # Do not fail the quote if persistence fails
        pass

    return PriceResponse(distance_km=d_km, quotes=quotes, cheapest=cheapest)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
