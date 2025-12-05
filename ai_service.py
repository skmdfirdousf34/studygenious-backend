import os
import json
import asyncio
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from models import (
    User, StudyPlan, Doubt, DoubtCreate, Question,
    Note, NoteCreate, Flashcard, TestAttempt,
    ExamType, Subject, QuestionDifficulty
)

load_dotenv()


class AIService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.client = AsyncOpenAI(api_key=self.api_key)

    # -----------------------------
    # INTERNAL AI CALL METHOD
    # -----------------------------
    async def _ask_ai(self, model: str, system: str, user_prompt: Any):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt}
        ]

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,
            temperature=0.2
        )

        return response.choices[0].message["content"]

    # =====================================================
    # 1. STUDY PLAN — uses GPT-4o-mini (cheap + smart)
    # =====================================================
    async def generate_study_plan(self, user: User, duration_days: int) -> StudyPlan:
        system = (
            "You are an expert JEE/NEET study planner. "
            "Generate strictly JSON output only."
        )

        user_prompt = f"""
Create a {duration_days}-day study plan for:
Class: {user.class_type}
Target: {user.exam_target}
Hours/day: {user.hours_available}
Weak: {user.weak_chapters}
Strong: {user.strong_chapters}
Days left: {user.days_left}

Return ONLY JSON with:
weekly_schedule, revision_dates, test_dates, focus_chapters, recommendations
"""

        try:
            res = await self._ask_ai("gpt-4o-mini", system, user_prompt)
            data = json.loads(res)
        except:
            data = {
                "weekly_schedule": [],
                "revision_dates": [],
                "test_dates": [],
                "focus_chapters": user.weak_chapters[:5],
                "recommendations": "Focus on weak chapters."
            }

        return StudyPlan(
            user_id=user.id,
            duration_days=duration_days,
            **data
        )

    # =====================================================
    # 2. DOUBT SOLVER — uses GPT-O1 for math, GPT-4o-mini for image
    # =====================================================
    async def solve_doubt(self, user: User, doubt_data: DoubtCreate) -> Doubt:
        exam_type = "JEE" if "JEE" in user.exam_target.value else "NEET"

        system = (
            f"You are an expert {exam_type} teacher. "
            "Provide detailed solution with steps, tricks, concepts."
        )

        # If image → Vision model
        if doubt_data.question_image:
            user_prompt = [
                {"type": "text", "text": "Solve this question with full explanation."},
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{doubt_data.question_image}"
                }
            ]

            model = "gpt-4o-mini"  # vision supported
        else:
            user_prompt = doubt_data.question_text

            # Math → gpt-o1
            if any(x in doubt_data.question_text.lower() for x in ["math", "integral", "limit", "physics", "force"]):
                model = "gpt-o1"
            else:
                model = "gpt-4o-mini"

        try:
            response = await self._ask_ai(model, system, user_prompt)
        except Exception as e:
            response = f"Error solving doubt: {e}"

        return Doubt(
            user_id=user.id,
            question_text=doubt_data.question_text,
            question_image=doubt_data.question_image,
            step_by_step=response,
            speed_trick="See explanation above",
            concept="Included",
            common_errors="Included",
            alternative_method="Included",
            exam_reference=exam_type,
            difficulty=QuestionDifficulty.MEDIUM,
            chapter="Auto-detected"
        )

    # =====================================================
    # 3. TEST PERFORMANCE — GPT-4o-mini
    # =====================================================
    async def analyze_test_performance(self, test: TestAttempt, questions, user_answers, score):
        system = "You analyze mock tests. Output strictly valid JSON."

        total = len(questions)
        correct = sum(1 for q in questions if user_answers.get(q["id"]) == q["correct_option"])
        accuracy = (correct / total * 100) if total else 0

        topics = [q.get("chapter", "Unknown") for q in questions]

        user_prompt = f"""
Analyze performance:
Total Q: {total}
Correct: {correct}
Score: {score}/{test.total_marks}
Accuracy: {accuracy}

Topics: {topics}

Return JSON with:
speed_analysis, weak_topics, strong_topics, recommendations
"""

        try:
            res = await self._ask_ai("gpt-4o-mini", system, user_prompt)
            data = json.loads(res)
            data["accuracy"] = accuracy
        except:
            data = {
                "accuracy": accuracy,
                "speed_analysis": "Need more tests.",
                "weak_topics": list(set(topics))[:3],
                "strong_topics": [],
                "recommendations": "Improve accuracy."
            }

        return data

    # =====================================================
    # 4. NOTES — GPT-4o-mini
    # =====================================================
    async def generate_notes(self, user, note_data):
        system = "Convert text into structured exam notes. Output JSON only."

        prompt = f"""
Text:
{note_data.original_text}

Return JSON with:
short_notes, exam_points, diagrams, example_problems
"""

        try:
            res = await self._ask_ai("gpt-4o-mini", system, prompt)
            data = json.loads(res)
        except:
            data = {
                "short_notes": "Error generating notes.",
                "exam_points": [],
                "diagrams": "",
                "example_problems": ""
            }

        return Note(
            user_id=user.id,
            title=note_data.title,
            original_text=note_data.original_text,
            **data,
            subject=note_data.subject,
            chapter=note_data.chapter
        )

    # =====================================================
    # 5. FLASHCARDS — GPT-4o-mini
    # =====================================================
    async def generate_flashcards(self, user: User, topic: str, count: int):
        system = "Generate flashcards. Output JSON array only."

        prompt = f"""
Topic: {topic}
Count: {count}

Each flashcard:
question, answer, flashcard_type
"""

        try:
            res = await self._ask_ai("gpt-4o-mini", system, prompt)
            data = json.loads(res)
        except:
            return []

        return [
            Flashcard(
                user_id=user.id,
                question=c.get("question", ""),
                answer=c.get("answer", ""),
                flashcard_type=c.get("flashcard_type", "QA")
            )
            for c in data
        ]

    # =====================================================
    # 6. MOTIVATION — GPT-4o-mini
    # =====================================================
    async def generate_motivation(self, user: User):
        system = "You are a motivational coach. Keep message under 100 words."

        prompt = f"""
Target: {user.exam_target}
Days left: {user.days_left}
Hours: {user.study_hours}
Tests: {user.tests_taken}
Doubts: {user.total_doubts}
"""

        try:
            return await self._ask_ai("gpt-4o-mini", system, prompt)
        except:
            return "Keep going! You're closer than you think."

