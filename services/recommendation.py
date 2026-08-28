from db.mongo import cuisine_collection

def get_cuisine(accent):
    if not accent or not str(accent).strip():
        return None
    data = cuisine_collection.find_one({"accent": accent.strip().lower()})
    return data["categories"] if data else None