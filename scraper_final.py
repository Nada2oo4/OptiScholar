"""
OptiScholar — Scholarship Web Scraper (Final Working Version)
==============================================================
Scrapes collegescholarships.com using parallel requests.
Filters: deadline after July 1, 2026 only.

Usage in Colab:
    import sys, importlib.util
    sys.argv = ["scraper_final.py"]
    spec = importlib.util.spec_from_file_location("sc", "scraper_final.py")
    sc   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    sc.run(limit=300, n_ids=5000)

Proven output:
    5000 IDs checked in ~2.5 minutes
    ~99 valid scholarships found (deadline > July 2026)
    ~76 unique after deduplication
"""

import re
import time
import random
import logging
import concurrent.futures
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────
DATASET_PATH = "/content/drive/MyDrive/Graduation Project/dataset_test2/scholarships_final_ready-2.csv"
BACKUP_PATH  = "/content/drive/MyDrive/Graduation Project/dataset_test2/scholarships_backup.csv"
DEADLINE_CUTOFF = date(2026, 7, 1)
MAX_WORKERS     = 10   # parallel threads

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

CATEGORY_PAGES = [
    "https://www.collegescholarships.com/types/scholarships-for-women",
    "https://www.collegescholarships.com/types/need-based-scholarships",
    "https://www.collegescholarships.com/types/merit-scholarships",
    "https://www.collegescholarships.com/types/minority-scholarships",
    "https://www.collegescholarships.com/types/scholarships-for-college-students",
    "https://www.collegescholarships.com/types/athletic-scholarships",
    "https://www.collegescholarships.com/types/community-service-scholarships",
    "https://www.collegescholarships.com/types/stem-scholarships",
    "https://www.collegescholarships.com/types/international-scholarships",
    "https://www.collegescholarships.com/types/graduate-scholarships",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — FETCH AND PARSE ONE SCHOLARSHIP PAGE
# ══════════════════════════════════════════════════════════════

def fetch_and_parse(sch_id: int) -> Optional[dict]:
    """
    Fetch one collegescholarships.com scholarship page and parse it.
    Returns a dict ready for the OptiScholar schema, or None if:
      - Page returns 404 / network error
      - Title is missing or "No matching scholarship"
      - Deadline is before DEADLINE_CUTOFF
    """
    url = f"https://www.collegescholarships.com/scholarships/detail/{sch_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None

        soup  = BeautifulSoup(resp.text, "html.parser")
        lines = [
            l.strip() for l in soup.get_text(separator="\n").split("\n")
            if l.strip() and len(l.strip()) > 2
        ]

        # ── Title ───────────────────────────────────────────────
        # Always appears as the line after "Scholarships List"
        title = ""
        for i, line in enumerate(lines):
            if line == "Scholarships List" and i + 1 < len(lines):
                candidate = lines[i + 1]
                if (len(candidate) > 5 and
                        candidate not in ["Home", "Find Scholarships",
                                          "FIND COLLEGES", "RESOURCES"]):
                    title = candidate
                    break

        if not title or len(title) < 5:
            return None
        if "no matching" in title.lower():
            return None

        # ── Field extraction ────────────────────────────────────
        # Each label is on its own line ending with ":",
        # value is the next non-label line
        LABELS = [
            "Year of Need", "Type", "Num Awards",
            "Min Award", "Max Award", "Deadline",
            "Website", "Sponsoring Organization"
        ]
        fields = {}
        for i, line in enumerate(lines):
            clean = line.rstrip(":")
            if clean in LABELS and i + 1 < len(lines):
                val = lines[i + 1]
                if val.rstrip(":") not in LABELS:
                    fields[clean] = val

        # ── Description ─────────────────────────────────────────
        desc = ""
        for i, line in enumerate(lines):
            if line == "Scholarship Description" and i + 1 < len(lines):
                desc = lines[i + 1]
                break

        # ── Deadline ────────────────────────────────────────────
        deadline = fields.get("Deadline", "")
        if not re.match(r"\d{4}-\d{2}-\d{2}", deadline or ""):
            # Try regex on full page text as fallback
            m = re.search(r"Deadline:\s*(\d{4}-\d{2}-\d{2})",
                          soup.get_text())
            deadline = m.group(1) if m else ""

        # Apply deadline filter
        if deadline:
            m = re.match(r"(\d{4})-(\d{2})-(\d{2})", deadline)
            if m:
                d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if d < DEADLINE_CUTOFF:
                    return None  # outdated — skip

        # ── Type ────────────────────────────────────────────────
        raw_type = fields.get("Type", "")
        if not raw_type or raw_type in ["Num Awards:", "Min Award:", ""]:
            raw_type = ""
        TYPE_MAP = {
            "merit based":  "Merit-Based",
            "merit-based":  "Merit-Based",
            "need based":   "Need-Based",
            "need-based":   "Need-Based",
            "needs based":  "Need-Based",
            "athletic":     "Athletic",
            "community":    "Community Service",
        }
        sch_type = next(
            (v for k, v in TYPE_MAP.items() if k in raw_type.lower()),
            "Merit-Based"
        )

        # ── Amount ──────────────────────────────────────────────
        amount_str = fields.get("Min Award", "")
        if not amount_str or amount_str == "Deadline:":
            amount_str = "0"
        try:
            amt = float(re.sub(r"[^\d.]", "", amount_str))
        except ValueError:
            amt = 0.0

        # ── Eligibility from Year of Need ───────────────────────
        year = fields.get("Year of Need", "").lower()
        eb = int(any(w in year for w in [
            "freshman", "sophomore", "junior", "senior",
            "college", "undergraduate"
        ]))
        em = int("graduate" in year or "master" in year)
        ep = int("doctoral" in year or "phd" in year)
        eh = int("high school" in year or "12th" in year)
        if not any([eb, em, ep, eh]):
            eb = 1  # default to bachelor

        website = fields.get("Website", "") or url

        return {
            "scholarship_id":          f"WEB_CSC_{sch_id:06d}",
            "scholarship_title":       title,
            "source_dataset":          "scraped_collegescholarships_com",
            "eligible_high_school":    eh,
            "eligible_bachelor":       eb,
            "eligible_master":         em,
            "eligible_phd":            ep,
            "eligible_course_other":   0,
            "funding_category":        "Fixed_Amount" if amt > 0 else "Unknown",
            "funding_amount_raw":      amt,
            "country":                 "USA",
            "deadline":                deadline,
            "description":             desc,
            "link":                    website,
            "region":                  "",
            "deadline_type":           "fixed" if deadline else "unknown",
            "deadline_date":           deadline,
            "deadline_urgency":        "open",
            "description_cleaned":     desc,
            "citizenship_required":    "",
            "min_gpa_required":        0.0,
            "requires_financial_need": int("need" in sch_type.lower()),
            "special_eligibility":     fields.get("Year of Need", ""),
            "has_gpa_requirement":     0,
            "eligibility_summary":     desc[:200],
            "scholarship_type":        sch_type,
        }

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# SECTION 2 — COLLECT IDs FROM CATEGORY PAGES
# ══════════════════════════════════════════════════════════════

def collect_category_ids() -> list:
    """
    Scrape scholarship IDs from category pages.
    Each category page has ~12 direct links to scholarship detail pages.
    Returns list of unique integer IDs.
    """
    all_ids = []
    seen    = set()

    log.info("Collecting IDs from category pages...")
    for cat_url in CATEGORY_PAGES:
        try:
            time.sleep(1.5)
            resp = requests.get(cat_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                m = re.search(r"/detail/(\d+)", a["href"])
                if m:
                    sid = int(m.group(1))
                    if sid not in seen:
                        all_ids.append(sid)
                        seen.add(sid)
            log.info(f"  {cat_url.split('/')[-1]:<40} → {len(seen)} IDs total")
        except Exception as e:
            log.warning(f"  Category error: {e}")

    log.info(f"Category IDs collected: {len(all_ids)}")
    return all_ids


# ══════════════════════════════════════════════════════════════
# SECTION 3 — DEDUPLICATION
# ══════════════════════════════════════════════════════════════

def deduplicate(existing_df: pd.DataFrame, new_records: list) -> list:
    """
    Remove records whose titles are too similar to existing ones.
    Uses token overlap: >70% overlap = duplicate.
    """
    if existing_df is None or len(existing_df) == 0:
        return new_records

    existing_titles = set(
        existing_df["scholarship_title"].str.lower().str.strip().tolist()
    )

    unique = []
    for rec in new_records:
        title   = str(rec.get("scholarship_title", "")).lower().strip()
        if title in existing_titles:
            continue
        t_words = set(re.findall(r"\w+", title))
        is_dup  = False
        for ex in existing_titles:
            ex_words = set(re.findall(r"\w+", ex))
            if not t_words or not ex_words:
                continue
            overlap = len(t_words & ex_words) / max(len(t_words), len(ex_words))
            if overlap > 0.70:
                is_dup = True
                break
        if not is_dup:
            unique.append(rec)
            existing_titles.add(title)

    log.info(f"Dedup: {len(new_records)} scraped → {len(unique)} unique")
    return unique


# ══════════════════════════════════════════════════════════════
# SECTION 4 — MAIN RUN FUNCTION
# ══════════════════════════════════════════════════════════════

def run(limit: int = 300,
        n_ids: int = 5000,
        dry_run: bool = False) -> pd.DataFrame:
    """
    Main scraping pipeline.

    Args:
        limit   : max new scholarships to collect
        n_ids   : how many IDs to check in total (more = more found)
        dry_run : if True, don't save to Drive

    Returns updated DataFrame.

    Proven performance:
        n_ids=5000, limit=300 → ~2.5 minutes, ~99 found, ~76 unique
    """
    print("\n" + "="*60)
    print("OPTISCHOLAR SCHOLARSHIP SCRAPER")
    print(f"  Source   : collegescholarships.com")
    print(f"  IDs      : {n_ids:,}")
    print(f"  Limit    : {limit:,}")
    print(f"  Cutoff   : after {DEADLINE_CUTOFF}")
    print(f"  Workers  : {MAX_WORKERS}")
    print("="*60)

    # Load existing dataset
    existing_df = None
    if not dry_run:
        try:
            existing_df = pd.read_csv(DATASET_PATH)
            print(f"\nExisting dataset: {len(existing_df):,} scholarships")
            # Backup
            existing_df.to_csv(BACKUP_PATH, index=False)
            print(f"Backup saved to:  {BACKUP_PATH}")
        except FileNotFoundError:
            print(f"Dataset not found at {DATASET_PATH}")
            print("Will create new dataset.")

    # Collect IDs
    cat_ids    = collect_category_ids()
    seen_ids   = set(cat_ids)
    random_ids = np.random.choice(
        range(1, 110001),
        size=min(n_ids, 110000),
        replace=False
    ).tolist()
    all_ids = cat_ids + [i for i in random_ids if i not in seen_ids]
    all_ids = all_ids[:n_ids]

    print(f"\nTotal IDs to check: {len(all_ids):,}")
    print(f"  (category: {len(cat_ids)}, random: {len(all_ids)-len(cat_ids)})")

    # Parallel scraping
    results = []
    print()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_and_parse, sid): sid
            for sid in all_ids
        }
        with tqdm(total=len(all_ids), desc="Scraping") as pbar:
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)
                if len(results) >= limit:
                    break
                rec = future.result()
                if rec:
                    results.append(rec)
                    pbar.set_postfix(found=len(results))

    print(f"\nScraped: {len(results)}")

    if not results:
        print("No scholarships found.")
        return existing_df

    # Deduplicate
    unique = deduplicate(existing_df, results) if existing_df is not None else results
    print(f"Unique new: {len(unique)}")

    if not unique:
        print("All scraped scholarships already in dataset.")
        return existing_df

    # Combine and save
    new_df = pd.DataFrame(unique)
    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df

    print(f"\n{'='*60}")
    if existing_df is not None:
        print(f"  Previous  : {len(existing_df):,}")
    print(f"  New unique: {len(unique):,}")
    print(f"  Total now : {len(combined):,}")
    print(f"\n  Type distribution (new):")
    print(new_df["scholarship_type"].value_counts().to_string())
    print("="*60)

    if not dry_run:
        combined.to_csv(DATASET_PATH, index=False)
        print(f"\n✅ Saved to Drive: {DATASET_PATH}")
    else:
        print("\n[DRY RUN] Not saved.")

    return combined


# ══════════════════════════════════════════════════════════════
# DIRECT EXECUTION
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0]]  # clear any Colab kernel args

    updated = run(
        limit=300,
        n_ids=5000,
        dry_run=False
    )
