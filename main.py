import os
import re
import uuid
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


load_dotenv(override=True)


def _clean_env_value(value: Optional[str]) -> str:
    if value is None:
        return ""
    return value.strip().strip('"').strip("'")


GROQ_API_KEY = _clean_env_value(os.getenv("GROQ_API_KEY"))
GROQ_MODEL = _clean_env_value(os.getenv("GROQ_MODEL")) or "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in .env file.")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


MAX_RESPONSE_WORDS = 180

RESPONSE_LENGTH_RULE = (
    f"STRICT OUTPUT LENGTH RULE: Your final answer must be {MAX_RESPONSE_WORDS} words or fewer. "
    "Never exceed 180 words. Do not add long introductions. Prefer concise bullets."
)


SYSTEM_PROMPT = f"""
{RESPONSE_LENGTH_RULE}

You are CareerCounsellorAI, an India-focused engineering career counselling agent for JEE and engineering entrance aspirants.

Your users are Indian Class 12 students, JEE Main/JEE Advanced aspirants, private engineering entrance aspirants, and parents.

Core job:
Help users choose realistic colleges and branches using rank, percentile, category, home state, gender, budget, affordability, city/location preference, preferred branch, placement/package expectations, and exams given.

You must help with:
1. JEE Main/JEE Advanced rank, percentile, score, and college possibilities
2. IIT, NIT, IIIT, GFTI, state, and private college comparisons
3. Branch choice: CSE, AI/ML, Data Science, ECE, EE, Mechanical, Civil, Chemical, Biotechnology, etc.
4. College-vs-branch tradeoffs
5. JoSAA, CSAB, state counselling, and private counselling strategy
6. Backup options when JEE result is not strong
7. Fees, affordability, placements, internship ecosystem, location, and ROI

Critical rules:
- Do not guarantee admission.
- Do not invent cutoffs, fees, placements, ranks, packages, rankings, or counselling dates.
- For current or college-specific facts, use web_search or jee_counselling_search before answering.
- Prefer official sources: JoSAA, CSAB, NTA, IIT/NIT/IIIT/GFTI official sites, state counselling sites, college official pages, NIRF, and official placement/fee PDFs.
- Cutoffs change every year. Always say recommendations are based on previous trends and should be verified during current counselling.
- If data is insufficient, say so clearly and ask exactly one useful follow-up question.

When recommending colleges:
- Respect every user constraint: rank/percentile, category, home state, gender, branch, fees/budget/affordability, preferred city/state, exam, and package/placement expectations.
- Give buckets: Ambitious, Moderate, Safe, and Backup when possible.
- For budget-sensitive users, prefer lower-fee government/state options first, then private options with clear ROI caution.
- For city-specific users, prioritize that city/nearby region, but mention if expanding location improves options.
- For package-focused users, do not blindly chase average package. Balance branch, placement consistency, alumni network, internship location, and fees.
- For weak rank/percentile, suggest realistic state/private counselling and branch flexibility.

Answer style:
- Final answer must be 180 words or fewer.
- Be crisp, practical, and empathetic.
- Use 3-6 bullets when useful.
- Prefer college names plus reason, not generic advice.
- Avoid long tables unless the user asks.
- If search results are long, summarize only the most useful college names and reasons.
- End with one next-step question only when information is missing.
"""


COMMON_CATEGORIES = {
    "general": "General",
    "gen": "General",
    "open": "General",
    "ews": "EWS",
    "obc": "OBC-NCL",
    "obc-ncl": "OBC-NCL",
    "ncl": "OBC-NCL",
    "sc": "SC",
    "st": "ST",
    "pwd": "PwD",
    "ews-pwd": "EWS PwD",
    "obc pwd": "OBC-NCL PwD",
    "sc pwd": "SC PwD",
    "st pwd": "ST PwD",
}


BRANCH_KEYWORDS = {
    "cse": "Computer Science Engineering / CSE",
    "computer science": "Computer Science Engineering / CSE",
    "cs": "Computer Science Engineering / CSE",
    "it": "Information Technology",
    "information technology": "Information Technology",
    "ai": "Artificial Intelligence",
    "aiml": "AI/ML",
    "ai/ml": "AI/ML",
    "artificial intelligence": "Artificial Intelligence",
    "machine learning": "Machine Learning",
    "data science": "Data Science",
    "ds": "Data Science",
    "ece": "Electronics and Communication Engineering",
    "electronics": "Electronics and Communication Engineering",
    "eee": "Electrical and Electronics Engineering",
    "electrical": "Electrical Engineering",
    "ee": "Electrical Engineering",
    "mechanical": "Mechanical Engineering",
    "mech": "Mechanical Engineering",
    "civil": "Civil Engineering",
    "chemical": "Chemical Engineering",
    "biotech": "Biotechnology",
    "biotechnology": "Biotechnology",
    "aerospace": "Aerospace Engineering",
    "production": "Production Engineering",
    "metallurgy": "Metallurgical Engineering",
    "instrumentation": "Instrumentation Engineering",
}


EXAM_KEYWORDS = {
    "jee main": "JEE Main",
    "jee mains": "JEE Main",
    "mains": "JEE Main",
    "jee advanced": "JEE Advanced",
    "advanced": "JEE Advanced",
    "bitsat": "BITSAT",
    "viteee": "VITEEE",
    "srmjee": "SRMJEEE",
    "comedk": "COMEDK",
    "mht cet": "MHT-CET",
    "mht-cet": "MHT-CET",
    "kcet": "KCET",
    "wbjee": "WBJEE",
    "jac delhi": "JAC Delhi",
    "ipu": "IPU CET / GGSIPU",
    "aktu": "AKTU / UPTAC",
    "uptac": "AKTU / UPTAC",
    "gujcet": "GUJCET",
    "tnea": "TNEA",
    "eamcet": "EAMCET",
    "ap eamcet": "AP EAMCET",
    "ts eamcet": "TS EAMCET",
    "keam": "KEAM",
}


COMMON_CITIES = [
    "Delhi", "New Delhi", "Noida", "Greater Noida", "Ghaziabad", "Gurgaon", "Gurugram",
    "Faridabad", "Mumbai", "Navi Mumbai", "Pune", "Nagpur", "Bengaluru", "Bangalore",
    "Hyderabad", "Chennai", "Coimbatore", "Kolkata", "Howrah", "Ahmedabad", "Gandhinagar",
    "Surat", "Vadodara", "Jaipur", "Jodhpur", "Kota", "Lucknow", "Kanpur", "Varanasi",
    "Prayagraj", "Allahabad", "Patna", "Ranchi", "Bhubaneswar", "Cuttack", "Bhopal",
    "Indore", "Jabalpur", "Gwalior", "Raipur", "Chandigarh", "Mohali", "Panchkula",
    "Dehradun", "Roorkee", "Haridwar", "Jalandhar", "Ludhiana", "Amritsar", "Kurukshetra",
    "Sonipat", "Panipat", "Hisar", "Kochi", "Thiruvananthapuram", "Trivandrum", "Kozhikode",
    "Mysuru", "Mysore", "Mangalore", "Manipal", "Vellore", "Trichy", "Tiruchirappalli",
    "Madurai", "Visakhapatnam", "Vijayawada", "Guntur", "Warangal", "Tirupati", "Goa",
    "Pilani", "Rourkela", "Durgapur", "Jamshedpur", "Silchar", "Guwahati", "Shillong",
    "Agartala", "Srinagar", "Jammu", "Hamirpur", "Mandi", "Surathkal", "Calicut",
]


COMMON_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Jammu and Kashmir", "Ladakh",
    "Puducherry", "Chandigarh",
]


class ChatRequest(BaseModel):
    question: str = Field(..., description="Question asked by student or parent")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID to continue previous conversation",
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class ResetRequest(BaseModel):
    session_id: str


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clean_snippet(text: str, max_chars: int = 280) -> str:
    text = normalize_spaces(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def money_to_lakh(amount: float, unit: str) -> float:
    unit = unit.lower().strip()
    if unit in {"cr", "crore", "crores"}:
        return amount * 100
    if unit in {"k", "thousand"}:
        return amount / 100
    return amount


def extract_fee_budget_lakh(text: str) -> Dict[str, Any]:
    lower = text.lower()
    budget: Dict[str, Any] = {}

    range_match = re.search(
        r"(?:fees?|budget|afford(?:able|ability)?|cost)?\s*(?:range\s*)?(?:₹|rs\.?|inr)?\s*"
        r"(\d+(?:\.\d+)?)\s*(?:-|to|–|—)\s*(?:₹|rs\.?|inr)?\s*"
        r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|l|cr|crore|crores)",
        lower,
    )
    if range_match:
        low = money_to_lakh(float(range_match.group(1)), range_match.group(3))
        high = money_to_lakh(float(range_match.group(2)), range_match.group(3))
        budget["fee_min_lakh"] = min(low, high)
        budget["fee_max_lakh"] = max(low, high)

    cap_match = re.search(
        r"(?:under|below|within|upto|up to|less than|max(?:imum)?|budget(?: is)?|fees? under)\s*"
        r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|l|cr|crore|crores)",
        lower,
    )
    if cap_match:
        budget["fee_max_lakh"] = money_to_lakh(float(cap_match.group(1)), cap_match.group(2))

    amount_match = re.search(
        r"(?:budget|fees?|afford(?:ability|able)?)\s*(?:is|=|:)?\s*(?:₹|rs\.?|inr)?\s*"
        r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|l|cr|crore|crores)",
        lower,
    )
    if amount_match and "fee_max_lakh" not in budget:
        budget["fee_max_lakh"] = money_to_lakh(float(amount_match.group(1)), amount_match.group(2))

    if any(word in lower for word in ["affordable", "low budget", "cheap", "less fees", "low fees", "economical", "roi"]):
        budget["affordability"] = "Budget-sensitive / ROI-focused"

    return budget


def find_known_terms(text: str, terms: List[str]) -> List[str]:
    lower = text.lower()
    found: List[str] = []
    seen: Set[str] = set()

    for term in sorted(terms, key=len, reverse=True):
        pattern = r"(?<![a-z])" + re.escape(term.lower()) + r"(?![a-z])"
        if re.search(pattern, lower) and term.lower() not in seen:
            found.append(term)
            seen.add(term.lower())

    return found


def extract_rank(text: str) -> Dict[str, Any]:
    lower = text.lower()
    result: Dict[str, Any] = {}

    rank_patterns = [
        r"\b(?:jee\s*main\s*)?(?:crl|air|rank)\s*(?:is|=|:|-)?\s*([0-9][0-9,]{1,8})\b",
        r"\b([0-9][0-9,]{1,8})\s*(?:crl|air|rank)\b",
    ]

    for pattern in rank_patterns:
        match = re.search(pattern, lower)
        if match:
            result["rank"] = int(match.group(1).replace(",", ""))
            break

    category_rank_match = re.search(
        r"\b(?:category\s*rank|cat\s*rank|obc\s*rank|ews\s*rank|sc\s*rank|st\s*rank)\s*"
        r"(?:is|=|:|-)?\s*([0-9][0-9,]{1,8})\b",
        lower,
    )
    if category_rank_match:
        result["category_rank"] = int(category_rank_match.group(1).replace(",", ""))

    percentile_match = re.search(r"\b(\d{1,2}(?:\.\d+)?|100(?:\.0+)?)\s*(?:%|percentile)\b", lower)
    if percentile_match:
        result["percentile"] = float(percentile_match.group(1))

    return result


def extract_category(text: str) -> Optional[str]:
    lower = text.lower().replace("_", "-")

    for key in sorted(COMMON_CATEGORIES, key=len, reverse=True):
        pattern = r"(?<![a-z])" + re.escape(key) + r"(?![a-z])"
        if re.search(pattern, lower):
            return COMMON_CATEGORIES[key]

    return None


def extract_gender(text: str) -> Optional[str]:
    lower = text.lower()

    if re.search(r"\b(female|girl|woman|women)\b", lower):
        return "Female"
    if re.search(r"\b(male|boy|man)\b", lower):
        return "Male"

    return None


def extract_branches(text: str) -> List[str]:
    lower = text.lower()
    branches: List[str] = []
    seen: Set[str] = set()

    for key, label in sorted(BRANCH_KEYWORDS.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = r"(?<![a-z])" + re.escape(key) + r"(?![a-z])"
        if re.search(pattern, lower) and label not in seen:
            branches.append(label)
            seen.add(label)

    return branches


def extract_exams(text: str) -> List[str]:
    lower = text.lower()
    exams: List[str] = []
    seen: Set[str] = set()

    for key, label in sorted(EXAM_KEYWORDS.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = r"(?<![a-z])" + re.escape(key) + r"(?![a-z])"
        if re.search(pattern, lower) and label not in seen:
            exams.append(label)
            seen.add(label)

    if not exams and any(word in lower for word in ["nit", "iiit", "gfti", "josaa", "csab"]):
        exams.append("JEE Main")

    if not exams and any(word in lower for word in ["iit", "jee"]):
        exams.append("JEE Main / JEE Advanced")

    return exams


def extract_package_lpa(text: str) -> Dict[str, Any]:
    lower = text.lower()
    package: Dict[str, Any] = {}

    package_patterns = [
        r"(?:package|placement|avg|average|median|ctc)\s*(?:of|above|more than|at least|min(?:imum)?|is|=|:)?\s*"
        r"(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?\s*pa|lakh\s*per\s*annum)",
        r"(\d+(?:\.\d+)?)\s*(?:lpa|lakhs?\s*pa|lakh\s*per\s*annum)\s*(?:package|placement|ctc)?",
    ]

    for pattern in package_patterns:
        package_match = re.search(pattern, lower)
        if package_match:
            package["target_package_lpa"] = float(package_match.group(1))
            break

    if any(word in lower for word in ["high package", "best package", "good package", "placement", "placements", "roi"]):
        package["placement_focus"] = True

    return package


def extract_student_preferences(text: str) -> Dict[str, Any]:
    text = normalize_spaces(text)
    preferences: Dict[str, Any] = {}

    preferences.update(extract_rank(text))

    category = extract_category(text)
    if category:
        preferences["category"] = category

    gender = extract_gender(text)
    if gender:
        preferences["gender"] = gender

    branches = extract_branches(text)
    if branches:
        preferences["branches"] = branches

    exams = extract_exams(text)
    if exams:
        preferences["exams"] = exams

    cities = find_known_terms(text, COMMON_CITIES)
    if cities:
        preferences["cities"] = cities[:4]

    states = find_known_terms(text, COMMON_STATES)
    if states:
        preferences["states"] = states[:3]

    preferences.update(extract_fee_budget_lakh(text))
    preferences.update(extract_package_lpa(text))

    lower = text.lower()

    if any(word in lower for word in ["private", "pvt", "deemed"]):
        preferences["college_type"] = "Private/deemed acceptable"

    if any(word in lower for word in ["government", "govt", "nit", "iiit", "gfti", "iit"]):
        preferences["college_type"] = "Government / centrally funded preferred"

    if any(word in lower for word in ["hostel", "hostel fees"]):
        preferences["hostel_needed"] = True

    return preferences


def profile_to_text(preferences: Dict[str, Any]) -> str:
    if not preferences:
        return (
            "No structured constraints were confidently extracted. "
            "Ask for missing rank/percentile, category, home state, branch, budget, and location if needed."
        )

    labels = {
        "rank": "CRL/AIR rank",
        "category_rank": "category rank",
        "percentile": "percentile",
        "category": "category",
        "gender": "gender",
        "branches": "preferred branches",
        "exams": "exams",
        "cities": "preferred cities",
        "states": "preferred states/home-state clues",
        "fee_min_lakh": "minimum fee budget in lakh INR",
        "fee_max_lakh": "maximum fee budget in lakh INR",
        "affordability": "affordability preference",
        "target_package_lpa": "target package in LPA",
        "placement_focus": "placement/package focus",
        "college_type": "college type preference",
        "hostel_needed": "hostel consideration",
    }

    lines = []

    for key, value in preferences.items():
        label = labels.get(key, key.replace("_", " "))
        if isinstance(value, list):
            value_text = ", ".join(map(str, value))
        else:
            value_text = str(value)
        lines.append(f"- {label}: {value_text}")

    return "\n".join(lines)


def build_constraint_phrase(preferences: Dict[str, Any]) -> str:
    parts: List[str] = []

    if preferences.get("rank"):
        parts.append(f"rank {preferences['rank']}")

    if preferences.get("category_rank"):
        parts.append(f"category rank {preferences['category_rank']}")

    if preferences.get("percentile"):
        parts.append(f"{preferences['percentile']} percentile")

    if preferences.get("category"):
        parts.append(str(preferences["category"]))

    if preferences.get("gender"):
        parts.append(str(preferences["gender"]))

    if preferences.get("branches"):
        parts.append(" ".join(preferences["branches"][:3]))

    if preferences.get("cities"):
        parts.append(" ".join(preferences["cities"][:3]))

    if preferences.get("states"):
        parts.append(" ".join(preferences["states"][:2]))

    if preferences.get("fee_max_lakh"):
        parts.append(f"fees under {preferences['fee_max_lakh']:g} lakh")

    if preferences.get("target_package_lpa"):
        parts.append(f"average package {preferences['target_package_lpa']:g} LPA")

    if preferences.get("exams"):
        parts.append(" ".join(preferences["exams"][:3]))

    return " ".join(parts)


def build_search_queries(student_query: str, preferences: Dict[str, Any]) -> List[str]:
    query = normalize_spaces(student_query)
    constraint_phrase = build_constraint_phrase(preferences)
    base = normalize_spaces(f"{query} {constraint_phrase}")

    queries: List[str] = []

    exams = " ".join(preferences.get("exams", [])) or "JEE Main JoSAA CSAB"
    branches = " ".join(preferences.get("branches", [])) or "engineering"
    cities = " ".join(preferences.get("cities", []))
    states = " ".join(preferences.get("states", []))
    category = preferences.get("category", "")
    rank = preferences.get("rank") or preferences.get("category_rank") or ""
    fee_cap = preferences.get("fee_max_lakh")
    target_package = preferences.get("target_package_lpa")

    if rank or any(exam in exams for exam in ["JEE Main", "JEE Main / JEE Advanced"]):
        queries.extend(
            [
                f"{base} JoSAA opening closing rank cutoff previous year official",
                f"{base} CSAB special round closing rank cutoff previous year",
                f"site:josaa.nic.in {rank} {category} {branches} opening closing rank",
                f"site:csab.nic.in {rank} {category} {branches} closing rank",
            ]
        )

    if cities or states:
        location = normalize_spaces(f"{cities} {states}")
        queries.extend(
            [
                f"best engineering colleges {location} {branches} fees placements official",
                f"{location} engineering colleges {branches} cutoff fees average package",
            ]
        )

    if fee_cap or preferences.get("affordability"):
        budget_text = f"fees under {fee_cap:g} lakh" if fee_cap else "affordable low fees ROI"
        queries.extend(
            [
                f"{budget_text} engineering colleges {cities or states} {branches} placements",
                f"government engineering colleges {states or cities} {branches} low fees placements",
            ]
        )

    if target_package or preferences.get("placement_focus"):
        package_text = f"average package {target_package:g} LPA" if target_package else "best placements average package"
        queries.extend(
            [
                f"{branches} engineering colleges {cities or states} {package_text} official placement report",
                f"{base} placement statistics average median package official",
            ]
        )

    if "Private/deemed acceptable" in str(preferences.get("college_type", "")):
        queries.append(f"private engineering colleges {cities or states} {branches} fees placements cutoff")

    queries.extend(
        [
            f"{base} college prediction realistic options",
            f"{base} fees placements cutoff official",
        ]
    )

    deduped: List[str] = []
    seen: Set[str] = set()

    for item in queries:
        item = normalize_spaces(item)
        key = item.lower()
        if len(item) >= 10 and key not in seen:
            deduped.append(item)
            seen.add(key)

    return deduped[:8]


def format_search_results(query: str, results: List[dict]) -> str:
    if not results:
        return f"No search results found for query: {query}"

    output = [f"Search query: {query}\n"]
    seen_urls: Set[str] = set()
    rank = 1

    for result in results:
        title = result.get("title", "No title")
        href = result.get("href") or result.get("url") or result.get("link", "No URL")
        body = result.get("body") or result.get("snippet", "")

        if href in seen_urls:
            continue

        seen_urls.add(href)

        output.append(
            f"{rank}. {clean_snippet(title, 140)}\n"
            f"URL: {href}\n"
            f"Snippet: {clean_snippet(body)}\n"
        )

        rank += 1

    return "\n".join(output)


def duckduckgo_text_search(query: str, max_results: int) -> List[dict]:
    with DDGS() as ddgs:
        return list(
            ddgs.text(
                query,
                region="in-en",
                safesearch="moderate",
                max_results=max_results,
            )
        )


@tool("web_search")
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the public web using DuckDuckGo.

    Use this for current information, latest college data, fees, placements,
    exam dates, counselling updates, branch comparisons, rankings, and news.
    Prefer official college/counselling/government/NIRF sources in the final answer.
    """
    try:
        max_results = max(1, min(int(max_results), 8))
        results = duckduckgo_text_search(query, max_results)
        return format_search_results(query, results)

    except Exception as e:
        return f"Web search failed for query '{query}'. Error: {str(e)}"


@tool("jee_counselling_search")
def jee_counselling_search(student_query: str, max_results_per_query: int = 4) -> str:
    """
    Targeted Indian engineering counselling search.

    Use this for JEE score/rank/percentile, college prediction, JoSAA, CSAB,
    NIT, IIIT, IIT, GFTI, state/private counselling, branch choice, previous-year
    cutoffs, opening/closing rank, fees, placements, city preference, affordability,
    and package expectations.

    The tool extracts constraints such as rank, category, budget, city, branch,
    and package expectation from the user's text, then searches multiple focused queries.
    """
    preferences = extract_student_preferences(student_query)
    queries = build_search_queries(student_query, preferences)
    max_results_per_query = max(1, min(int(max_results_per_query), 6))

    outputs = [
        "Parsed student constraints:",
        profile_to_text(preferences),
        "",
        "Use these constraints while recommending colleges. Do not ignore fee, city, branch, rank, category, or package preferences if present.",
    ]

    try:
        for query in queries:
            results = duckduckgo_text_search(query, max_results_per_query)
            outputs.append("\n" + "=" * 80 + "\n")
            outputs.append(format_search_results(query, results))

        return "\n".join(outputs)

    except Exception as e:
        return f"JEE counselling search failed. Error: {str(e)}"


def build_agent():
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.15,
        max_retries=2,
        groq_api_key=GROQ_API_KEY,
    )

    tools = [
        web_search,
        jee_counselling_search,
    ]

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


def extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_parts = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                if "text" in block:
                    text_parts.append(str(block["text"]))
                elif block.get("type") == "text" and "content" in block:
                    text_parts.append(str(block["content"]))
                elif isinstance(block.get("content"), str):
                    text_parts.append(block["content"])

            else:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    text_parts.append(block_text)

                block_content = getattr(block, "content", None)
                if isinstance(block_content, str):
                    text_parts.append(block_content)

        return "\n".join(part for part in text_parts if part).strip()

    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return text_attr.strip()

    content_attr = getattr(content, "content", None)
    if isinstance(content_attr, str):
        return content_attr.strip()

    return str(content).strip()


def build_enriched_question(question: str) -> str:
    preferences = extract_student_preferences(question)

    return (
        f"{RESPONSE_LENGTH_RULE}\n\n"
        f"Student question:\n{question}\n\n"
        f"Parsed constraints from this message:\n{profile_to_text(preferences)}\n\n"
        "Instruction: Use all parsed constraints. "
        "If recommending colleges, search first and bucket options as Ambitious, Moderate, Safe, and Backup when possible. "
        "The final answer sent to the user must be 180 words or fewer. "
        "If a constraint may be missing, ask only one follow-up question."
    )


app = FastAPI(
    title="CareerCounsellorAI API",
    description="FastAPI backend for Indian engineering counselling using Groq llama-3.1-8b-instant.",
    version="2.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


agent = build_agent()


chat_sessions: Dict[str, List[Dict[str, str]]] = {}


@app.get("/")
def root():
    return {
        "message": "CareerCounsellorAI API is running with Groq",
        "provider": "groq",
        "model": GROQ_MODEL,
        "docs": "/docs",
        "chat_endpoint": "/chat",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": GROQ_MODEL,
        "provider": "groq",
    }


@app.get("/debug/groq-config")
def debug_groq_config():
    return {
        "provider": "groq",
        "model": GROQ_MODEL,
        "api_key_present": bool(GROQ_API_KEY),
        "api_key_prefix": GROQ_API_KEY[:8] if GROQ_API_KEY else None,
        "api_key_length": len(GROQ_API_KEY),
    }


@app.get("/debug/groq-test")
def debug_groq_test():
    try:
        test_llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
            max_retries=0,
            groq_api_key=GROQ_API_KEY,
        )

        response = test_llm.invoke("Reply with only OK")

        return {
            "status": "ok",
            "model": GROQ_MODEL,
            "response": extract_text_from_content(response.content),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq test failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    chat_history = chat_sessions[session_id]

    enriched_question = build_enriched_question(question)

    messages = chat_history + [
        {
            "role": "user",
            "content": enriched_question,
        }
    ]

    try:
        result = await run_in_threadpool(
            agent.invoke,
            {"messages": messages},
        )

        final_message = result["messages"][-1]
        answer = extract_text_from_content(final_message.content)

        if not answer:
            answer = (
                "I could not generate a text response. Please try again with rank/percentile, "
                "category, home state, branch, and budget."
            )

        chat_sessions[session_id].append(
            {
                "role": "user",
                "content": question,
            }
        )

        chat_sessions[session_id].append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent failed: {str(e)}",
        )


@app.post("/reset")
def reset_session(request: ResetRequest):
    if request.session_id in chat_sessions:
        del chat_sessions[request.session_id]

    return {
        "status": "reset",
        "session_id": request.session_id,
    }


@app.get("/sessions")
def list_sessions():
    return {
        "total_sessions": len(chat_sessions),
        "session_ids": list(chat_sessions.keys()),
    }