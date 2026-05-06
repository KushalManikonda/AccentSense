from db.mongo import cuisine_collection

def get_cuisine(accent):
    data = cuisine_collection.find_one({"accent": accent.lower()})
    return data["categories"] if data else None