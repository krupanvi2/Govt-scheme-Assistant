# app.py  |  run: streamlit run app.py
# .env file: LLM_PROVIDER=gemini  GEMINI_API_KEY=xxxx  (or groq / ollama)

import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import tldextract
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

load_dotenv()

# ---------------- DATA ----------------
SCHEMES = [
    dict(name="PM-KISAN", category="Agriculture", description="Income support for small and marginal farmer families.",
         eligibility="Landholding farmer families (exclusions apply).", benefits="Rs 6000 per year in three instalments.",
         how_to_apply="Register on PM-KISAN portal or CSC with land records and Aadhaar.", url="https://pmkisan.gov.in",
         min_age=18, max_age=100, gender="any", max_income=0, occupation="farmer"),
    dict(name="Ayushman Bharat PM-JAY", category="Health", description="Public health insurance for poor and vulnerable families.",
         eligibility="Families listed in SECC database or notified categories.", benefits="Health cover up to Rs 5 lakh per family per year.",
         how_to_apply="Check eligibility on PM-JAY portal or at an empanelled hospital.", url="https://pmjay.gov.in",
         min_age=0, max_age=100, gender="any", max_income=0, occupation="any"),
    dict(name="Sukanya Samriddhi Yojana", category="Savings / Girl Child", description="Savings scheme for a girl child's education and marriage.",
         eligibility="Girl child below 10 years; opened by parent or guardian.", benefits="High-interest savings with Section 80C tax benefit.",
         how_to_apply="Open at a post office or authorised bank with birth certificate and guardian ID.", url="https://www.nsiindia.gov.in",
         min_age=0, max_age=10, gender="female", max_income=0, occupation="any"),
    dict(name="PM Ujjwala Yojana", category="Energy / Women", description="LPG connections for women from poor households.",
         eligibility="Adult woman from BPL or eligible household without LPG connection.", benefits="Deposit-free LPG connection.",
         how_to_apply="Apply at LPG distributor or PMUY portal with KYC and ration card.", url="https://www.pmuy.gov.in",
         min_age=18, max_age=100, gender="female", max_income=0, occupation="any"),
    dict(name="Atal Pension Yojana", category="Pension", description="Guaranteed pension for unorganised sector workers.",
         eligibility="Indian citizen aged 18-40 with a savings bank account.", benefits="Monthly pension of Rs 1000-5000 after age 60.",
         how_to_apply="Apply via your bank or post office with Aadhaar.", url="https://www.npscra.nsdl.co.in",
         min_age=18, max_age=40, gender="any", max_income=0, occupation="any"),
    dict(name="PM MUDRA Yojana", category="Business / Loans", description="Collateral-free loans for micro and small enterprises.",
         eligibility="Small business owners or those starting one.", benefits="Loans up to Rs 10 lakh (Shishu, Kishor, Tarun).",
         how_to_apply="Apply at banks, NBFCs or MFIs with business plan and KYC.", url="https://www.mudra.org.in",
         min_age=18, max_age=100, gender="any", max_income=0, occupation="entrepreneur"),
    dict(name="PM Jan Dhan Yojana", category="Banking", description="National mission for financial inclusion.",
         eligibility="Any Indian citizen aged 10+ without a bank account.", benefits="Zero-balance account, RuPay card, accident insurance.",
         how_to_apply="Visit any bank branch with Aadhaar or other ID.", url="https://pmjdy.gov.in",
         min_age=10, max_age=100, gender="any", max_income=0, occupation="any"),
    dict(name="PM Suraksha Bima Yojana", category="Insurance", description="Low-cost accident insurance.",
         eligibility="Age 18-70 with a savings bank account.", benefits="Rs 2 lakh for accidental death or total disability, about Rs 20 per year.",
         how_to_apply="Enrol through your bank.", url="https://jansuraksha.gov.in",
         min_age=18, max_age=70, gender="any", max_income=0, occupation="any"),
    dict(name="National Scholarship Portal", category="Education", description="Single window for central and state scholarships.",
         eligibility="Students meeting each scholarship's criteria.", benefits="Amount varies by scheme.",
         how_to_apply="Register on NSP with marks, income certificate and Aadhaar.", url="https://scholarships.gov.in",
         min_age=10, max_age=35, gender="any", max_income=0, occupation="student"),
    dict(name="Stand-Up India", category="Business / Loans", description="Bank loans for women and SC/ST first-time entrepreneurs.",
         eligibility="Women or SC/ST entrepreneurs above 18 starting a greenfield enterprise.", benefits="Loan of Rs 10 lakh to Rs 1 crore.",
         how_to_apply="Apply via Stand-Up India portal or a bank branch.", url="https://www.standupmitra.in",
         min_age=18, max_age=100, gender="any", max_income=0, occupation="entrepreneur"),
    dict(name="PM Awas Yojana", category="Housing", description="Housing assistance for urban and rural families.",
         eligibility="Families without a pucca house (EWS/LIG/rural criteria).", benefits="Financial assistance or interest subsidy for a house.",
         how_to_apply="Apply via PMAY portal, municipal office or bank.", url="https://pmaymis.gov.in",
         min_age=18, max_age=100, gender="any", max_income=0, occupation="any"),
]

# ---------------- DATABASE ----------------
engine = create_engine("sqlite:///chat_history.db")


class Base(DeclarativeBase):
    pass


class ChatLog(Base):
    __tablename__ = "chat_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    schemes: Mapped[str] = mapped_column(String(300), default="")
    created: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)


def save_chat(q, a, s=""):
    with Session(engine) as ses:
        ses.add(ChatLog(question=q, answer=a, schemes=s))
        ses.commit()


def recent_chats(n=5):
    with Session(engine) as ses:
        return ses.scalars(select(ChatLog).order_by(ChatLog.id.desc()).limit(n)).all()


# ---------------- LLM ----------------
SYSTEM_PROMPT = ("You explain Indian government schemes in simple language. Answer ONLY from the context given. "
                 "If the answer is not in the context, say so and suggest the official website. "
                 "Be concise; mention eligibility, benefits and how to apply.")


def ask_llm(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    try:
        if provider == "gemini":
            from google import genai
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            r = client.models.generate_content(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                                               contents=f"{SYSTEM_PROMPT}\n\n{prompt}")
            return r.text
        if provider == "groq":
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"},
                              json={"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                                    "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                                 {"role": "user", "content": prompt}], "temperature": 0.2},
                              timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        if provider == "ollama":
            r = requests.post(os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
                              json={"model": os.getenv("OLLAMA_MODEL", "llama3.2"),
                                    "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}", "stream": False}, timeout=120)
            r.raise_for_status()
            return r.json()["response"]
        return "Set LLM_PROVIDER to gemini, groq or ollama in .env"
    except Exception as e:
        return f"LLM error: {e}"


# ---------------- SEARCH ----------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_index():
    df = pd.DataFrame(SCHEMES)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = (df["name"] + ". " + df["category"] + ". " + df["description"] + " Eligibility: "
            + df["eligibility"] + " Benefits: " + df["benefits"]).tolist()
    return df, model, model.encode(docs, normalize_embeddings=True)


def search(query, k=3):
    df, model, emb = get_index()
    scores = cosine_similarity(model.encode([query], normalize_embeddings=True), emb)[0]
    return df.iloc[np.argsort(scores)[::-1][:k]]


def context_text(rows):
    return "\n\n".join(f"Scheme: {r['name']}\nAbout: {r['description']}\nEligibility: {r['eligibility']}\n"
                       f"Benefits: {r['benefits']}\nHow to apply: {r['how_to_apply']}\nSite: {r['url']}"
                       for _, r in rows.iterrows())


def eligible(df, age, gender, occupation, income):
    m = (df["min_age"] <= age) & (df["max_age"] >= age)
    m &= df["gender"].isin(["any", gender.lower()])
    m &= df["occupation"].isin(["any", occupation.lower()])
    m &= (df["max_income"] == 0) | (df["max_income"] >= income)
    return df[m]


# ---------------- SCRAPER ----------------
def fetch_official_page(url, max_chars=6000):
    ext = tldextract.extract(url)
    if ext.suffix not in ("gov.in", "nic.in"):
        raise ValueError("Only official .gov.in / .nic.in links are allowed.")
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for t in soup(["script", "style", "nav", "footer", "header"]):
        t.decompose()
    return " ".join(soup.get_text(" ").split())[:max_chars]


# ---------------- UI ----------------
st.set_page_config(page_title="Govt Scheme Assistant", page_icon="🇮🇳", layout="wide")
st.title("🇮🇳 AI Government Scheme Assistant")
st.caption("Always verify details on the official website.")

df, _, _ = get_index()

with st.sidebar:
    st.header("Your profile")
    age = st.number_input("Age", 0, 100, 20)
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    occupation = st.selectbox("Occupation", ["Student", "Farmer", "Entrepreneur", "Salaried", "Homemaker", "Other"])
    income = st.number_input("Annual family income (Rs)", 0, 10_000_000, 200000, step=10000)
    st.divider()
    st.header("Recent questions")
    for c in recent_chats():
        st.caption(f"• {c.question[:60]}")

tab1, tab2, tab3 = st.tabs(["💬 Ask", "✅ Schemes for me", "🌐 Read official page"])

with tab1:
    st.session_state.setdefault("messages", [])
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("e.g. Which scheme gives loans to start a business?")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        hits = search(q)
        prompt = (f"User profile: age {age}, {gender}, {occupation}, income Rs {income}.\n\n"
                  f"Context:\n{context_text(hits)}\n\nQuestion: {q}")
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = ask_llm(prompt)
            st.markdown(ans)
            st.caption("Sources: " + ", ".join(hits["name"]))
        st.session_state.messages.append({"role": "assistant", "content": ans})
        save_chat(q, ans, ", ".join(hits["name"]))

with tab2:
    matches = eligible(df, age, gender, occupation, income)
    st.subheader(f"{len(matches)} scheme(s) match your profile")
    for _, r in matches.iterrows():
        with st.expander(f"{r['name']} · {r['category']}"):
            st.write(r["description"])
            st.markdown(f"**Eligibility:** {r['eligibility']}")
            st.markdown(f"**Benefits:** {r['benefits']}")
            st.markdown(f"**How to apply:** {r['how_to_apply']}")
            st.link_button("Official website", r["url"])

with tab3:
    url = st.text_input("Official .gov.in / .nic.in scheme page URL")
    question = st.text_input("Your question", "Summarise eligibility, benefits and how to apply.")
    if st.button("Fetch & answer") and url:
        try:
            text = fetch_official_page(url)
            st.markdown(ask_llm(f"Context:\n{text}\n\nQuestion: {question}"))
        except Exception as e:
            st.error(str(e))