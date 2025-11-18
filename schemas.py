"""
Database Schemas for PriceFix

Each Pydantic model represents a MongoDB collection. The collection name is the
lowercased class name.
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional


class User(BaseModel):
    """
    Users collection schema
    Collection name: "user"
    """
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    password_hash: str = Field(..., description="Hashed password")
    auth_provider: str = Field("password", description="'password' or 'google'")
    avatar_url: Optional[str] = Field(None, description="Avatar URL if available")
    is_active: bool = Field(True, description="Whether user is active")


class RideQuery(BaseModel):
    """
    Ride queries collection schema
    Collection name: "ridequery"
    """
    user_email: Optional[EmailStr] = Field(None, description="Email of the requester")
    origin: str = Field(..., description="Origin address or place name")
    destination: str = Field(..., description="Destination address or place name")
    distance_km: float = Field(..., ge=0, description="Computed distance in kilometers")
    cheapest_provider: str = Field(..., description="Provider name with lowest price")
    cheapest_price: float = Field(..., ge=0, description="Cheapest quoted price")
