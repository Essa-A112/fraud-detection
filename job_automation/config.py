import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# ── API keys (loaded from .env, never hardcoded) ───────────────────────────────
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_SERVICE_ACCOUNT_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_PATH")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")

# ── Job search preferences ─────────────────────────────────────────────────────
JOB_TITLES = [
    "Data Analyst",
    "Data Scientist",
    "Data Engineer",
    "AI Engineer",
    "ML Engineer",
]

LOCATION = "London, UK"

# Salary filter: jobs must advertise at least this annual figure (GBP)
MIN_SALARY = 30000

# Seniority keywords that are acceptable (case-insensitive match against job title/description)
ACCEPTED_SENIORITY = ["graduate", "junior", "entry", "entry-level", "associate"]

# Set to True to skip jobs that are not fully remote
REMOTE_ONLY = False

# ── Scheduler ─────────────────────────────────────────────────────────────────
# Daily run time in 24-hour format "HH:MM"
DAILY_RUN_TIME = "08:00"

# ── Candidate details (used by Claude for CV tailoring and cover letters) ──────
CANDIDATE_NAME = "Essa Abikar"
CANDIDATE_PHONE = "07412 474466"
CANDIDATE_EMAIL = "essaabikar12@gmail.com"
CANDIDATE_LINKEDIN = "linkedin.com/in/essaabikar"

CANDIDATE_CV = """Essa Abikar
07412 474466 — essaabikar12@gmail.com — linkedin.com/in/essaabikar

Data Scientist & Analytics Engineer

EDUCATION

King's College London                                                  Sept 2024 – Sept 2025
MSc in Robotics (Merit)
- Relevant modules: Machine Learning; Neural Networks & Deep Learning; Systems & Control;
  Robot Kinematics; Sensing & Perception.

City, University of London                                             Sept 2020 – May 2024
BEng in Mechanical Engineering (Upper second-class honours)

RESEARCH & ACADEMIC PROJECTS

AI-Powered Investment Decision Support for UK Care Homes              Sept 2024 – Sept 2025
Master's Thesis
- Built an end-to-end ML pipeline on UK public datasets (ONS, CQC, Land Registry):
  ingestion, cleaning, reconciliation, feature engineering, and modelling.
- Trained and evaluated models (Random Forest, XGBoost, CatBoost) and produced
  decision-ready outputs with clear validation and sensitivity checks.
- Delivered an interactive Streamlit dashboard with explainability (SHAP/LIME) and
  LLM-driven Q&A to communicate insights to non-technical users.

Engineers Without Borders MSc Project                                  Oct 2024 – May 2025
IoT Water Station for Rural Communities
- Designed a data-informed service concept for a solar-powered IoT water kiosk;
  translated stakeholder needs into measurable requirements and system KPIs.
- Built cost and operating assumptions to compare design options; presented trade-offs
  clearly to support group decisions under time constraints.
- Evaluated sustainability impacts using LCA and systems thinking; produced concise
  visuals and recommendations for a non-technical audience.

International Conference Paper                                         Sept 2024 – Sept 2025
"Evaluating Compressor Selection and Performance in High-Temperature Heat Pump Systems
for Industrial Applications"
- Developed and validated models against empirical data; produced publication-quality
  results quantifying efficiency and reliability trade-offs.
- Compared configurations (single-/two-stage, cascade) and synthesised findings into
  defensible recommendations for industrial applications.

PROFESSIONAL EXPERIENCE

Kirfu                                                                  Dec 2023 – June 2025
Technical Analyst (AI-Logistics Start-Up)
- Supported last-mile optimisation work using geospatial and operational data; cleaned
  datasets, defined KPIs, and delivered weekly insight summaries for stakeholders.
- Improved reliability of recurring Python analyses by standardising inputs, debugging
  scripts, and documenting assumptions to make outputs repeatable.
- Conducted and synthesised 50+ driver interviews into a quantified operational case for
  funding; contributed to securing £4,500 from the City Star Project.

Royal Mail                                                             Feb 2022 – Jan 2025
Delivery Driver
- Maintained a 97% completion rate while operating under time and quality constraints;
  prioritised workload dynamically across high-volume routes.
- Trained and mentored 10 new starters, improving on-boarding speed and reducing
  operational errors through clear process guidance.
- Introduced route planning habits that reduced wasted travel and improved daily
  throughput during peak periods.

Haya Community Centre                                                  Apr 2018 – Present
Community Volunteer
- Planned and delivered community events; coordinated logistics and communicated with
  diverse groups to ensure safe, well-run sessions.

SKILLS

Data & Analytics:   SQL, data cleaning, feature engineering, EDA, KPI design,
                    statistical analysis
Python:             Pandas, NumPy, scikit-learn, XGBoost; reproducible notebooks
                    and scripts
ML & Evaluation:    classification/regression, cross-validation, error analysis,
                    bias checks
Explainability:     SHAP, LIME; stakeholder-ready model narratives
Dashboards:         Streamlit; translating analysis into self-serve tools
Engineering Tools:  MATLAB, CoolProp; systems modelling fundamentals
Workflow:           GitHub, documentation, versioned outputs
Soft Skills:        stakeholder communication, presentations, structured
                    problem-solving, mentoring
"""

COVER_LETTER_TONE_SAMPLE = """Essa Abikar
07412 474466 — essaabikar12@gmail.com — linkedin.com/in/essaabikar

Dear Hiring Manager,

I am writing to apply for the Private Equity Performance Improvement (PEPI) Work Experience
Programme at Alvarez & Marsal. I hold a BEng in Mechanical Engineering and I am completing
an MSc in Robotics at King's College London. Across my academic and professional experience,
I have developed a strong analytical foundation, with a growing focus on how data can be used
to evaluate performance and support investment decisions.

Through my recent work on an investment-focused project, I developed a strong interest in
how businesses are assessed and how value is created in practice. Working with large UK datasets,
I analysed regional performance, identified key drivers, and translated data into clear insights
to support decision-making. This experience made me realise that many investment decisions rely
on structured analysis, identifying trends, and understanding what drives performance—areas that
align closely with the work carried out within PEPI.

What attracts me to this opportunity is the ability to apply analytical thinking in a commercial
environment, supporting both pre-acquisition analysis and post-acquisition performance improvement.
I am particularly interested in understanding how companies are evaluated, where inefficiencies
exist, and how data-driven insights can support strategic decisions that directly impact value
creation across the investment lifecycle.

Alongside my academic work, my experience at an AI logistics start-up further strengthened my
ability to work with real-world data and operational challenges. I analysed delivery performance,
identified inefficiencies, and supported the development of optimisation strategies, contributing
to improved decision-making and a successful funding proposal. This experience developed my
ability to translate analysis into practical outcomes, work with stakeholders, and operate in
fast-paced environments.

I am keen to contribute, learn from experienced professionals, and gain hands-on exposure to
how analytical insights are applied in a commercial context. I see this programme as a valuable
opportunity to further develop my understanding while applying the analytical skills I have
built so far.

Thank you for considering my application.

Yours faithfully,
Essa Abikar
"""
