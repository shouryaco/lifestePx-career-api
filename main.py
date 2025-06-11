from fastapi import FastAPI, Form
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.lifestepx.com"],  # Use specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GuidanceRequest(BaseModel):
    name: str
    experience: int
    skills: str
    industry: str
    resume_text: str = ""

@app.post("/career-guidance")
def get_career_guidance(data: GuidanceRequest):
    prompt = f"""
You are an expert AI career coach. Use the following user details to give personalized career guidance.

Name: {data.name}
Experience: {data.experience} years
Skills: {data.skills}
Industry preference: {data.industry}

Resume:
{data.resume_text[:2000]}

Please provide:
1. Suggested job roles
2. Skill gaps & resources
3. Resume and interview tips
4. Job platforms
5. Motivational advice
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful and strategic career advisor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return {"advice": response.choices[0].message.content.strip()}
