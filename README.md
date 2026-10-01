# 🎓 OptiScholar

**Personalized Intelligent Multi-model Scholarship Recommendation System**

OptiScholar is an AI-powered scholarship recommendation system that matches students with relevant scholarships based on their profile, academic background, interests, and scholarship requirements. It combines Neural Collaborative Filtering with embedding-based transfer learning, NLP-based document parsing, SHAP explainability, and an LLM-powered chatbot, delivered through a Streamlit web application.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)

---

## 📌 Overview

Finding the right scholarship is slow and fragmented. Opportunities are spread across many sources, each with its own eligibility criteria, deadlines, and documents, and students have to read and compare them manually. Simple keyword or rule-based matching struggles here: it cannot learn which scholarships actually suit which kinds of students, and it gives no sense of *why* something was recommended.

OptiScholar addresses this with a multi-model pipeline:

- **Recommendation:** a Neural Collaborative Filtering (NCF) model, extended with a transfer network built on learned embeddings, ranks scholarships for each student.
- **NLP:** a fine-tuned RoBERTa NER model, combined with regex rules, extracts profile information from student documents. Semantic text-matching approaches (BiLSTM, CrossEncoder, SBERT) were compared during development.
- **Explainability:** SHAP is used to show which features drive a recommendation.
- **Chatbot:** an LLM assistant, served through the Groq API, helps students with scholarship questions using context from the application.

---

## 🎯 Objectives

- Provide personalized scholarship recommendations from a student's profile
- Move beyond rule-based matching to a learned recommendation model
- Make recommendations interpretable through explainability
- Reduce manual effort by extracting profile data from student documents
- Support students with an AI chatbot inside the application
- Collect scholarship data through a scraper

---

## ✨ Key Features

- **NCF recommendation pipeline** with a transfer network on top of learned embeddings
- **Student profile handling** with local persistence in SQLite
- **Document parsing** using a hybrid approach: fine-tuned RoBERTa NER plus regex
- **SHAP explainability** visualizations for model predictions
- **AI chatbot** powered by an LLM through the Groq API
- **Scholarship scraper** built with BeautifulSoup and Requests
- **Streamlit web interface** connecting all components

---

## 🧠 AI / ML Architecture

### 1. Data Collection
Scholarship data is gathered with a scraper (`scraper_final.py`) built on BeautifulSoup and Requests. The recommendation model is trained on a student–scholarship interaction dataset of about 20K raw records, which grew to roughly 29K rows after binarization and negative sampling. The data is split 80/20 with stratification, and class imbalance is handled with a weighted loss (`pos_weight`).

### 2. Recommendation Model
The core model is a Neural Collaborative Filtering network (`ncf_pipeline.py`) implemented in PyTorch. A transfer network operating on the learned embeddings produces the final scoring, and the trained weights are stored in `models/`.

### 3. NLP / Document Processing
`document_parser_hybrid.py` extracts structured profile fields from student documents by combining a fine-tuned RoBERTa NER model (Transformers) with regex patterns. During development, BiLSTM, CrossEncoder, and pretrained SBERT approaches were compared for semantic text matching.

### 4. Explainable AI
`shap_visualizations.py` uses SHAP to attribute a recommendation score to individual input features, so users can see what pushed a scholarship up or down the ranking.

### 5. AI Chatbot
`chatbot.py` implements an LLM assistant through the Groq API. It uses a context-injection approach: relevant application context is passed to the model in the prompt. It is not a retrieval-augmented (RAG) system.

### 6. Application Layer
`app.py` is the Streamlit entry point. It connects the profile input, database (`database.py`), recommendation pipeline, explainability views, and chatbot into a single interface.

---

## 🔄 System Workflow

1. **Student profile**: entered manually or extracted from an uploaded document
2. **Document parsing**: RoBERTa NER + regex extract profile fields
3. **Recommendation pipeline**: NCF + transfer network score scholarships
4. **Ranking**: scholarships are ordered by predicted relevance
5. **Explanation**: SHAP shows the features behind each recommendation
6. **Chatbot**: the student asks follow-up questions and gets LLM-generated answers

---

## 🏗️ Project Structure

```
OptiScholar/
├── app.py                      # Streamlit application entry point
├── chatbot.py                  # Groq-based LLM chatbot
├── database.py                 # SQLite database layer
├── document_parser_hybrid.py   # Hybrid document parser (RoBERTa NER + regex)
├── ncf_pipeline.py             # NCF recommendation pipeline
├── scraper_final.py            # Scholarship scraper
├── shap_visualizations.py      # SHAP explainability visualizations
├── requirements.txt
└── models/
    ├── ncf_model_v2.pt         # Trained NCF model weights
    └── transfer_fn_v2.pt       # Transfer network weights
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| ML | scikit-learn |
| NLP | Transformers (RoBERTa NER), SBERT |
| Recommendation | Neural Collaborative Filtering |
| Explainability | SHAP |
| LLM | Groq API |
| Database | SQLite |
| Web App | Streamlit |
| Data Processing | Pandas, NumPy |
| Web Scraping | BeautifulSoup, Requests |

<!-- TODO: before publishing, compare this table with requirements.txt and add/remove libraries (e.g. Matplotlib, PDF/OCR libraries such as PyMuPDF, pdfplumber, pdf2image, Tesseract) only if they are actually used. -->

---

## ⚙️ Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/Nada2oo4/OptiScholar.git
cd OptiScholar
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**3. Activate it**

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Configure environment variables**

Create a `.env` file in the project root and add your Groq API key:

```
GROQ_API_KEY=your_api_key_here
```

> `.env` is intentionally excluded from Git. Never commit API keys.

**6. Add local artifacts (not included in the repository)**

- Place your datasets in `data/`
- Place the fine-tuned RoBERTa NER model in `fine_tuned_roberta/`

See [Dataset & Model Files](#-dataset--model-files) for details.

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Then open the local URL that Streamlit prints in the terminal (by default `http://localhost:8501`).

---

## 📊 Dataset & Model Files

Some artifacts are **intentionally excluded** from this public repository. This is by design, not an omission.

| Excluded item | Reason |
|---|---|
| `data/` | CSV datasets are kept local and not published |
| `fine_tuned_roberta/` | Model weights are too large for normal GitHub storage |
| `optischolar.db` | Local database file |
| `Dissertation/` | Thesis files are not part of the code repository |
| `.env` | Contains secrets |
| `.venv/`, archives | Local environment and ZIP files |

**Datasets.** The CSV datasets live locally under `data/` and are not part of the repository.

**Fine-tuned NLP model.** The fine-tuned RoBERTa model is excluded because its weights are too large for a regular GitHub repository. The repository contains the implementation code that uses it, but not the weights.

The trained recommendation weights (`models/ncf_model_v2.pt`, `models/transfer_fn_v2.pt`) are included.

---

## 🔍 Explainability

SHAP is used to explain the recommendation model's predictions. Instead of returning only a ranked list, the application can show which input features contributed most to a scholarship's score and in which direction. This makes the model's behavior easier to inspect and helps users understand why a particular scholarship was suggested. SHAP explains the model's learned behavior; it is an interpretation aid, not a guarantee of causal reasoning.

---

## 💬 AI Chatbot

The chatbot helps students ask questions about scholarships and their recommendations inside the application.

- **Provider:** Groq API
- **Configured model:** `openai/gpt-oss-120b`
- **Approach:** context injection, where relevant application context is included in the prompt
- **Configuration:** the API key is loaded from the `GROQ_API_KEY` environment variable (via `.env`); no key is stored in the repository

---
## Demo
https://drive.google.com/file/d/1lcTkK7eQb8YwGobsg5zStsDUeT6CSw2F/view?usp=share_link

---
## 🖼️ Screenshots

<table>
  <tr>
    <td align="center"><b>Main Dashboard</b></td>
    <td align="center"><b>Scholarship Recommendations</b></td>
  </tr>
  <tr>
    <img width="1280" height="800" alt="Screenshot 2026-10-01 at 4 45 36 PM" src="https://github.com/user-attachments/assets/c26ffc0d-a4ee-429d-821e-4a91ff0755c3" />
<img width="1280" height="800" alt="Screenshot 2026-10-01 at 4 44 40 PM" src="https://github.com/user-attachments/assets/e107045e-0f1b-4ac8-82bc-4e76b359eecb" />

    
  </tr>

  <tr>
    <img width="1280" height="800" alt="Screenshot 2026-10-01 at 4 46 53 PM" src="https://github.com/user-attachments/assets/e1ac6b7b-d09a-49ea-90d0-a602ca193ace" />
<img width="1280" height="800" alt="Screenshot 2026-10-01 at 4 47 07 PM" src="https://github.com/user-attachments/assets/4cd1561c-557a-49d5-a701-153f0f82a54e" />
<img width="1280" height="800" alt="Screenshot 2026-10-01 at 4 47 25 PM" src="https://github.com/user-attachments/assets/96507699-2bb9-4ee8-9d50-b4cbad1b7160" />
<img width="1280" height="800" alt="Screenshot 2026-10-01 at 4 48 06 PM" src="https://github.com/user-attachments/assets/9eb31b58-f557-4d92-9a4b-80360f48eed7" />
<img width="1280" height="800" alt="Screenshot 2026-10-01 at 4 48 32 PM" src="https://github.com/user-attachments/assets/561a6b09-5875-47d9-ab66-b27feb58ee1b" />
<img width="1280" height="800" alt="Screenshot 2026-10-01 at 4 49 22 PM" src="https://github.com/user-attachments/assets/0d7591ff-844f-4da2-a672-90f52812b4ba" />
<img width="1280" height="800" alt="Screenshot 2026-10-01 at 4 49 51 PM" src="https://github.com/user-attachments/assets/cba649e5-4ddf-486f-b377-160e21e38c27" />
<img width="1280" height="800" alt="Screenshot 2026-10-01 at 4 46 37 PM" src="https://github.com/user-attachments/assets/e015fa77-fe96-4dd9-beed-c39dc7980007" />

    
  </tr>
</table>

---

## 📈 Results / Evaluation

| Model / Component | Metric | Result |
|---|---|---|
| Transfer Network (recommendation) | AUC | **0.8237** (project target: 0.800) |

**Setup:** ~20K raw interaction records processed to ~29K after binarization and negative sampling; 80/20 stratified train/test split; class imbalance handled with `pos_weight`. Models were trained on a Google Colab T4 GPU.

**NLP comparison:** BiLSTM, CrossEncoder, and pretrained SBERT were compared for semantic text matching.

<img width="1004" height="219" alt="Screenshot 2026-10-01 at 4 56 43 PM" src="https://github.com/user-attachments/assets/a408317b-d455-4512-a994-49193d97ffb3" />
<img width="450" height="343" alt="Screenshot 2026-10-01 at 4 57 00 PM" src="https://github.com/user-attachments/assets/c847a2f4-985c-494b-b1f8-9c21d5edf0d6" />
<img width="369" height="158" alt="Screenshot 2026-10-01 at 4 55 57 PM" src="https://github.com/user-attachments/assets/0d3b3649-dcc2-4ab4-a773-fc2059b39d31" />


---

##  Limitations

- Datasets and the fine-tuned RoBERTa model are not included in the repository, so the full pipeline cannot be reproduced from a fresh clone without supplying these artifacts locally.
- Recommendation quality depends on the size and coverage of the interaction and scholarship data.
- The chatbot depends on an external LLM API (Groq) and requires a valid API key and network access.
- The interaction data was processed with negative sampling, so results reflect that experimental setup.

---

##  Future Improvements

*These are planned directions, not implemented features.*

- Expand the scholarship and interaction data
- Improve personalization, for example through feedback-based learning
- Improve multilingual NLP support
- Add broader evaluation (more metrics, ablations, and user studies)
- Improve deployment and scalability

---

## 👩‍💻 Author

**Nada Ashraf**
AI Engineer | ML Engineer

- GitHub: [github.com/Nada2oo4](https://github.com/Nada2oo4)
- LinkedIn: _add your LinkedIn URL here_

---

## 📜 License

This project was developed as a graduation project. Please contact the author before reusing or redistributing the code.
