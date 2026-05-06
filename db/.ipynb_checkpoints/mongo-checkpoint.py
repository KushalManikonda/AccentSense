from pymongo import MongoClient

MONGO_URI = "mongodb+srv://kushalmanikonda_db_user:V3klf7MlSpOi1sUQ@accentsense.izufdym.mongodb.net/?appName=AccentSense"

client = MongoClient(MONGO_URI)
db = client["accentsense"]

cuisine_collection = db["cuisines"]
history_collection = db["accent_history"]
users_collection = db["users"]