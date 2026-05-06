import bcrypt
import uuid
from datetime import datetime
from db.mongo import users_collection

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_user(email, password, full_name):
    if users_collection.find_one({"email": email}):
        return False, "User already exists"

    user = {
        "username": email.split("@")[0] + "_" + str(uuid.uuid4())[:5],
        "email": email,
        "password": hash_password(password),
        "auth_provider": "local",
        "full_name": full_name,
        "is_verified": True,
        "roles": ["user"],
        "preferences": {"cuisine_types": []},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }

    result = users_collection.insert_one(user)
    user["_id"] = result.inserted_id

    return True, user

def login_user(email, password):
    user = users_collection.find_one({"email": email})

    if not user:
        return False, "User not found"

    if not verify_password(password, user["password"]):
        return False, "Invalid password"

    return True, user