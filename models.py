from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


# Enums
class ExamType(str, Enum):
    JEE_MAIN = "JEE_MAIN"
    JEE_ADVANCED = "JEE_ADVANCED"
    NEET = "NEET"


class ClassType(str, Enum):
    CLASS_11 = "CLASS_11"
    CLASS_12 = "CLASS_12"
    DROPPER = "DROPPER"


class StudyStyle(str, Enum):
    FAST_TRACK = "FAST_TRACK"
    BALANCED = "BALANCED"
    SLOW_PACED = "SLOW_PACED"


class SubscriptionTier(str, Enum):
    FREE = "FREE"
    BASIC = "BASIC"
    PREMIUM = "PREMIUM"
    PRO = "PRO"


class QuestionDifficulty(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class Subject(str, Enum):
    PHYSICS = "PHYSICS"
    CHEMISTRY = "CHEMISTRY"
    MATHEMATICS = "MATHEMATICS"
    BIOLOGY = "BIOLOGY"


# User Model
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    password_hash: str
    
    # Onboarding data
    class_type: Optional[ClassType] = None
    exam_target: Optional[ExamType] = None
    hours_available: Optional[int] = None
    coaching_hours: Optional[int] = None
    weak_chapters: List[str] = []
    strong_chapters: List[str] = []
    days_left: Optional[int] = None
    study_style: Optional[StudyStyle] = None
    sleep_cycle: Optional[str] = None
    preferred_timings: Optional[str] = None
    
    # Subscription
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    subscription_expiry: Optional[datetime] = None
    
    # Stats
    doubts_asked_today: int = 0
    total_doubts: int = 0
    tests_taken: int = 0
    study_hours: float = 0.0
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UserCreate(BaseModel):
    name: str
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class UserOnboarding(BaseModel):
    class_type: ClassType
    exam_target: ExamType
    hours_available: int
    coaching_hours: int
    weak_chapters: List[str]
    strong_chapters: List[str]
    days_left: int
    study_style: StudyStyle
    sleep_cycle: str
    preferred_timings: str


# Study Plan Model
class DailySchedule(BaseModel):
    time_slot: str
    activity: str
    subject: Optional[str] = None
    chapter: Optional[str] = None
    duration: int  # minutes


class StudyPlan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    duration_days: int  # 30, 60, 90, 180
    
    weekly_schedule: List[Dict[str, Any]] = []
    daily_schedules: List[DailySchedule] = []
    revision_dates: List[str] = []
    test_dates: List[str] = []
    
    focus_chapters: List[str] = []
    recommendations: str = ""
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# Doubt Model
class Doubt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    
    # Question data
    question_image: Optional[str] = None  # base64
    question_text: str = ""
    
    # AI Response
    step_by_step: str = ""
    speed_trick: str = ""
    concept: str = ""
    common_errors: str = ""
    alternative_method: str = ""
    exam_reference: str = ""
    difficulty: QuestionDifficulty = QuestionDifficulty.MEDIUM
    chapter: str = ""
    subtopic: str = ""
    
    # User actions
    saved: bool = False
    added_to_flashcards: bool = False
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class DoubtCreate(BaseModel):
    question_image: Optional[str] = None
    question_text: str = ""


# Question Bank Model
class Question(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    exam_type: ExamType
    subject: Subject
    chapter: str
    subtopic: Optional[str] = None
    
    question_text: str
    question_image: Optional[str] = None  # base64
    
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_option: str
    
    difficulty: QuestionDifficulty
    marks: int
    explanation: str
    
    pyq_year: Optional[int] = None
    is_pyq: bool = False
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Test Model
class TestAttempt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    test_type: str  # CHAPTER_WISE, FULL_TEST, MIXED
    
    exam_type: ExamType
    subject: Optional[Subject] = None
    chapter: Optional[str] = None
    
    questions: List[str] = []  # question IDs
    user_answers: Dict[str, str] = {}  # question_id: selected_option
    
    score: int = 0
    total_marks: int = 0
    time_taken: int = 0  # seconds
    
    # AI Analysis
    accuracy: float = 0.0
    speed_analysis: str = ""
    weak_topics: List[str] = []
    strong_topics: List[str] = []
    rank_prediction: Optional[int] = None
    recommendations: str = ""
    
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class TestSubmit(BaseModel):
    test_id: str
    user_answers: Dict[str, str]
    time_taken: int


# Notes Model
class Note(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    
    title: str
    original_text: str
    
    # AI Generated
    short_notes: str = ""
    exam_points: List[str] = []
    diagrams: str = ""
    example_problems: str = ""
    
    subject: Optional[Subject] = None
    chapter: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class NoteCreate(BaseModel):
    title: str
    original_text: str
    subject: Optional[Subject] = None
    chapter: Optional[str] = None


# Flashcard Model
class Flashcard(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    
    question: str
    answer: str
    flashcard_type: str  # QA, FORMULA, NCERT_LINE, ASSERTION_REASON, TRUE_FALSE
    
    subject: Optional[Subject] = None
    chapter: Optional[str] = None
    
    review_count: int = 0
    last_reviewed: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


# Study Tracker Model
class StudySession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    
    session_type: str  # STUDY, POMODORO, TEST
    subject: Optional[str] = None
    chapter: Optional[str] = None
    
    duration: int  # minutes
    focus_score: float = 0.0  # 0-100
    
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None


# Leaderboard Model
class LeaderboardEntry(BaseModel):
    user_id: str
    username: str
    score: int
    tests_taken: int
    study_hours: float
    rank: Optional[int] = None
    region: Optional[str] = None
