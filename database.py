import pymongo
import bcrypt
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB connection from environment variable
MONGODB_URI = os.getenv('MONGODB_URI')
client = pymongo.MongoClient(MONGODB_URI)
db = client['researchmate_db']
users_collection = db['users']

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def create_user(email, password, name):
    if users_collection.find_one({"email": email}):
        return False, "Email already exists"
    
    hashed_password = hash_password(password)
    user = {
        "email": email,
        "password": hashed_password,
        "name": name,
        "created_at": datetime.utcnow()
    }
    users_collection.insert_one(user)
    return True, "User created successfully"

def verify_user(email, password):
    user = users_collection.find_one({"email": email})
    if not user:
        return False, "User not found"
    
    if verify_password(password, user['password']):
        return True, user
    return False, "Incorrect password"

def update_password(email, new_password):
    hashed_password = hash_password(new_password)
    result = users_collection.update_one(
        {"email": email},
        {"$set": {"password": hashed_password}}
    )
    return result.modified_count > 0