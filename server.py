import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI, APIRouter, HTTPException, status
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from typing import List, Optional
import bcrypt
from datetime import datetime, timedelta
import asyncio

from models import (
    User, UserCreate, UserLogin, UserOnboarding,
    StudyPlan, Doubt, DoubtCreate, Question,
    TestAttempt, TestSubmit, Note, NoteCreate,
    Flashcard, StudySession, LeaderboardEntry,
    ExamType, ClassType, Subject, QuestionDifficulty, SubscriptionTier
)
from ai_service import AIService

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Initialize AI Service
ai_service = AIService()

# Create the main app
app = FastAPI(title="StudyGenius Pro API")
print("⭐⭐ CORS MIDDLEWARE LOADED ⭐⭐")

# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Allow all origins (frontend)
    allow_credentials=True,
    allow_methods=["*"],        # Allow all methods: GET, POST, etc.
    allow_headers=["*"],        # Allow all headers
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== AUTH ROUTES ====================

@api_router.post("/auth/signup", response_model=User)
async def signup(user_data: UserCreate):
    """Register a new user"""
    # Check if user exists
    existing = await db.users.find_one({"email": user_data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    password_hash = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # Create user
    user = User(
        name=user_data.name,
        email=user_data.email,
        password_hash=password_hash
    )
    
    await db.users.insert_one(user.dict())
    return user


@api_router.post("/auth/login")
async def login(credentials: UserLogin):
    """Login user"""
    user_doc = await db.users.find_one({"email": credentials.email})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = User(**user_doc)
    
    # Verify password
    if not bcrypt.checkpw(credentials.password.encode('utf-8'), user.password_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "user_id": user.id,
        "name": user.name,
        "email": user.email,
        "subscription_tier": user.subscription_tier,
        "onboarding_complete": user.exam_target is not None
    }


@api_router.post("/auth/onboarding/{user_id}")
async def complete_onboarding(user_id: str, data: UserOnboarding):
    """Complete user onboarding"""
    update_data = data.dict()
    update_data["updated_at"] = datetime.utcnow()
    
    result = await db.users.update_one(
        {"id": user_id},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "Onboarding completed", "user_id": user_id}


@api_router.get("/user/{user_id}", response_model=User)
async def get_user(user_id: str):
    """Get user profile"""
    user_doc = await db.users.find_one({"id": user_id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    return User(**user_doc)

app.include_router(api_router)


# ==================== STUDY PLANNER ROUTES ====================

@api_router.post("/planner/generate/{user_id}")
async def generate_study_plan(user_id: str, duration_days: int):
    """Generate AI study plan"""
    user_doc = await db.users.find_one({"id": user_id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = User(**user_doc)
    
    # Generate study plan using AI
    plan = await ai_service.generate_study_plan(user, duration_days)
    
    # Save to database
    await db.study_plans.insert_one(plan.dict())
    
    return plan


@api_router.get("/planner/{user_id}")
async def get_study_plans(user_id: str):
    """Get all study plans for user"""
    plans = await db.study_plans.find({"user_id": user_id}).sort("created_at", -1).to_list(100)
    return [StudyPlan(**plan) for plan in plans]


@api_router.get("/planner/{user_id}/current")
async def get_current_plan(user_id: str):
    """Get current active study plan"""
    plan_doc = await db.study_plans.find_one(
        {"user_id": user_id},
        sort=[("created_at", -1)]
    )
    if not plan_doc:
        raise HTTPException(status_code=404, detail="No study plan found")
    return StudyPlan(**plan_doc)


# ==================== DOUBT SOLVER ROUTES ====================

@api_router.post("/doubts/solve/{user_id}")
async def solve_doubt(user_id: str, doubt_data: DoubtCreate):
    """Solve a doubt using AI"""
    # Check rate limits
    user_doc = await db.users.find_one({"id": user_id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = User(**user_doc)
    
    # Rate limiting for free users
    if user.subscription_tier == SubscriptionTier.FREE:
        if user.doubts_asked_today >= 3:
            raise HTTPException(
                status_code=429,
                detail="Daily limit reached. Upgrade to Premium for unlimited doubts."
            )
    
    # Solve doubt using AI
    doubt = await ai_service.solve_doubt(user, doubt_data)
    
    # Save to database
    await db.doubts.insert_one(doubt.dict())
    
    # Update user stats
    await db.users.update_one(
        {"id": user_id},
        {
            "$inc": {"doubts_asked_today": 1, "total_doubts": 1},
            "$set": {"updated_at": datetime.utcnow()}
        }
    )
    
    return doubt


@api_router.get("/doubts/{user_id}")
async def get_doubts(user_id: str, saved_only: bool = False):
    """Get user's doubt history"""
    query = {"user_id": user_id}
    if saved_only:
        query["saved"] = True
    
    doubts = await db.doubts.find(query).sort("created_at", -1).to_list(100)
    return [Doubt(**doubt) for doubt in doubts]


@api_router.patch("/doubts/{doubt_id}/save")
async def save_doubt(doubt_id: str):
    """Toggle save status of a doubt"""
    doubt_doc = await db.doubts.find_one({"id": doubt_id})
    if not doubt_doc:
        raise HTTPException(status_code=404, detail="Doubt not found")
    
    new_status = not doubt_doc.get("saved", False)
    await db.doubts.update_one(
        {"id": doubt_id},
        {"$set": {"saved": new_status}}
    )
    
    return {"doubt_id": doubt_id, "saved": new_status}


# ==================== QUESTION BANK ROUTES ====================

@api_router.get("/questions")
async def get_questions(
    exam_type: Optional[ExamType] = None,
    subject: Optional[Subject] = None,
    chapter: Optional[str] = None,
    difficulty: Optional[QuestionDifficulty] = None,
    limit: int = 50
):
    """Get questions from question bank"""
    query = {}
    if exam_type:
        query["exam_type"] = exam_type
    if subject:
        query["subject"] = subject
    if chapter:
        query["chapter"] = chapter
    if difficulty:
        query["difficulty"] = difficulty
    
    questions = await db.questions.find(query).limit(limit).to_list(limit)
    return [Question(**q) for q in questions]


@api_router.post("/questions/seed")
async def seed_questions():
    """Seed sample questions (for development)"""
    # Check if already seeded
    count = await db.questions.count_documents({})
    if count > 0:
        return {"message": "Questions already seeded", "count": count}
    
    # Sample questions
    sample_questions = await ai_service.generate_sample_questions()
    
    if sample_questions:
        await db.questions.insert_many([q.dict() for q in sample_questions])
    
    return {"message": "Questions seeded", "count": len(sample_questions)}


# ==================== TEST ROUTES ====================

@api_router.post("/tests/create/{user_id}")
async def create_test(
    user_id: str,
    test_type: str,
    exam_type: ExamType,
    subject: Optional[Subject] = None,
    chapter: Optional[str] = None,
    question_count: int = 30
):
    """Create a new test"""
    # Get questions
    query = {"exam_type": exam_type}
    if subject:
        query["subject"] = subject
    if chapter:
        query["chapter"] = chapter
    
    questions = await db.questions.aggregate([
        {"$match": query},
        {"$sample": {"size": question_count}}
    ]).to_list(question_count)
    
    if not questions:
        raise HTTPException(status_code=404, detail="No questions found")
    
    # Calculate total marks
    total_marks = sum(q.get("marks", 4) for q in questions)
    
    # Create test attempt
    test = TestAttempt(
        user_id=user_id,
        test_type=test_type,
        exam_type=exam_type,
        subject=subject,
        chapter=chapter,
        questions=[q["id"] for q in questions],
        total_marks=total_marks
    )
    
    await db.test_attempts.insert_one(test.dict())
    
    # Return test with questions
    return {
        "test": test,
        "questions": [Question(**q) for q in questions]
    }


@api_router.post("/tests/submit")
async def submit_test(submission: TestSubmit):
    """Submit test and get AI analysis"""
    test_doc = await db.test_attempts.find_one({"id": submission.test_id})
    if not test_doc:
        raise HTTPException(status_code=404, detail="Test not found")
    
    test = TestAttempt(**test_doc)
    
    # Get questions
    questions = await db.questions.find(
        {"id": {"$in": test.questions}}
    ).to_list(len(test.questions))
    
    # Calculate score
    score = 0
    correct_answers = {}
    for q in questions:
        question = Question(**q)
        correct_answers[question.id] = question.correct_option
        if submission.user_answers.get(question.id) == question.correct_option:
            score += question.marks
    
    # AI Analysis
    analysis = await ai_service.analyze_test_performance(
        test, questions, submission.user_answers, score
    )
    
    # Update test
    await db.test_attempts.update_one(
        {"id": submission.test_id},
        {"$set": {
            "user_answers": submission.user_answers,
            "score": score,
            "time_taken": submission.time_taken,
            "accuracy": analysis.get("accuracy", 0),
            "speed_analysis": analysis.get("speed_analysis", ""),
            "weak_topics": analysis.get("weak_topics", []),
            "strong_topics": analysis.get("strong_topics", []),
            "recommendations": analysis.get("recommendations", ""),
            "completed_at": datetime.utcnow()
        }}
    )
    
    # Update user stats
    await db.users.update_one(
        {"id": test.user_id},
        {"$inc": {"tests_taken": 1}}
    )
    
    return {
        "score": score,
        "total_marks": test.total_marks,
        "correct_answers": correct_answers,
        "analysis": analysis
    }


@api_router.get("/tests/{user_id}/history")
async def get_test_history(user_id: str):
    """Get user's test history"""
    tests = await db.test_attempts.find(
        {"user_id": user_id, "completed_at": {"$ne": None}}
    ).sort("started_at", -1).to_list(100)
    return [TestAttempt(**test) for test in tests]


# ==================== NOTES MAKER ROUTES ====================

@api_router.post("/notes/create/{user_id}")
async def create_notes(user_id: str, note_data: NoteCreate):
    """Generate AI notes from text"""
    user_doc = await db.users.find_one({"id": user_id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = User(**user_doc)
    
    # Check subscription for premium feature
    if user.subscription_tier == SubscriptionTier.FREE:
        raise HTTPException(
            status_code=403,
            detail="Notes Maker is a Premium feature. Please upgrade."
        )
    
    # Generate notes using AI
    note = await ai_service.generate_notes(user, note_data)
    
    await db.notes.insert_one(note.dict())
    
    return note


@api_router.get("/notes/{user_id}")
async def get_notes(user_id: str):
    """Get user's notes"""
    notes = await db.notes.find({"user_id": user_id}).sort("created_at", -1).to_list(100)
    return [Note(**note) for note in notes]


# ==================== FLASHCARDS ROUTES ====================

@api_router.post("/flashcards/generate/{user_id}")
async def generate_flashcards(user_id: str, topic: str, count: int = 10):
    """Generate AI flashcards for a topic"""
    user_doc = await db.users.find_one({"id": user_id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = User(**user_doc)
    
    # Generate flashcards
    flashcards = await ai_service.generate_flashcards(user, topic, count)
    
    if flashcards:
        await db.flashcards.insert_many([f.dict() for f in flashcards])
    
    return flashcards


@api_router.get("/flashcards/{user_id}")
async def get_flashcards(user_id: str):
    """Get user's flashcards"""
    flashcards = await db.flashcards.find({"user_id": user_id}).to_list(500)
    return [Flashcard(**f) for f in flashcards]


# ==================== STUDY TRACKER ROUTES ====================

@api_router.post("/tracker/session")
async def log_study_session(session: StudySession):
    """Log a study session"""
    await db.study_sessions.insert_one(session.dict())
    
    # Update user study hours
    await db.users.update_one(
        {"id": session.user_id},
        {"$inc": {"study_hours": session.duration / 60.0}}
    )
    
    return {"message": "Session logged", "session_id": session.id}


@api_router.get("/tracker/{user_id}/stats")
async def get_study_stats(user_id: str, days: int = 7):
    """Get study statistics"""
    start_date = datetime.utcnow() - timedelta(days=days)
    
    sessions = await db.study_sessions.find({
        "user_id": user_id,
        "started_at": {"$gte": start_date}
    }).to_list(1000)
    
    total_duration = sum(s.get("duration", 0) for s in sessions)
    avg_focus = sum(s.get("focus_score", 0) for s in sessions) / len(sessions) if sessions else 0
    
    return {
        "total_sessions": len(sessions),
        "total_hours": total_duration / 60.0,
        "average_focus_score": avg_focus,
        "sessions": [StudySession(**s) for s in sessions]
    }


# ==================== LEADERBOARD ROUTES ====================

@api_router.get("/leaderboard/global")
async def get_global_leaderboard(limit: int = 50):
    """Get global leaderboard"""
    users = await db.users.aggregate([
        {"$project": {
            "user_id": "$id",
            "username": "$name",
            "tests_taken": 1,
            "study_hours": 1,
            "score": {"$add": [
                {"$multiply": ["$tests_taken", 10]},
                {"$multiply": ["$study_hours", 5]}
            ]}
        }},
        {"$sort": {"score": -1}},
        {"$limit": limit}
    ]).to_list(limit)
    
    # Add ranks
    for idx, user in enumerate(users, 1):
        user["rank"] = idx
    
    return users


# ==================== MOTIVATIONAL COACH ROUTES ====================

@api_router.get("/motivation/{user_id}/daily")
async def get_daily_motivation(user_id: str):
    """Get daily motivational message"""
    user_doc = await db.users.find_one({"id": user_id})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = User(**user_doc)
    message = await ai_service.generate_motivation(user)
    
    return {"message": message}


# ==================== SUBSCRIPTION ROUTES ====================

@api_router.post("/subscription/purchase/{user_id}")
async def purchase_subscription(user_id: str, tier: SubscriptionTier, duration_days: int = 30):
    """Mock subscription purchase"""
    expiry = datetime.utcnow() + timedelta(days=duration_days)
    
    result = await db.users.update_one(
        {"id": user_id},
        {"$set": {
            "subscription_tier": tier,
            "subscription_expiry": expiry,
            "updated_at": datetime.utcnow()
        }}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "message": "Subscription activated",
        "tier": tier,
        "expiry": expiry
    }


# ==================== HEALTH CHECK ====================

@api_router.get("/")
async def root():
    return {"message": "StudyGenius Pro API is running", "version": "1.0.0"}


@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
