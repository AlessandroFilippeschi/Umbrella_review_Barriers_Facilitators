#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

import tensorflow_hub as hub
import tensorflow as tf

# Use spaCy for lemmatization
import spacy
nlp = spacy.load("en_core_web_sm")

# Use HDBSCAN for clustering
import hdbscan

# Use RapidFuzz for fuzzy matching
from rapidfuzz import fuzz

########################################################################
## 1. Stopwords & Term Standardization Dictionaries
########################################################################
EXTENDED_STOPWORDS = {
    # Basic English stopwords
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "would", "should", "could", "ought",
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "to", "from", "of", "in",
    "on", "for", "at", "by", "about", "as", "into", "like", "through", "after", "over",
    "between", "out", "against", "during", "without", "before", "under", "around", "among",
    "same", "so", "than", "too", "very", "just", "can", "will", "of",
    # Domain filler words
    "lack", "limited", "issues", "issue", "barrier", "barriers", "concern"
    # Additional generic words
    "low", "challenge", "poor", "problem", "patient", "high", "higher", "requirement"
}
########################################################################
## Term Standardization Dictionary
########################################################################

TERM_STANDARDIZATION = {
    # 1. Technical / System / Software Issues
    "technical_issues": [
        "technical problems", "tech issues", "system errors", "software issues", "hardware issues",
        "technical challenges", "mechanical/software problems", "software glitches", "platform inadequacy",
        "technical accuracy limitations", "technical complexity", "incompatibility with existing systems",
        "bugs", "IT failures", "technical disruptions", "system malfunctions", "error messages",
        "technical faults", "debugging issues", "software crashes", "hardware failures",
        "Technical system errors", "Software audio delays", "Software video delays", "Technical glitches",
        "specific on-body placement of multiple biosensors", "Device synchronization problems",
        "Calibration problems", "Hardware dependencies", "Incorrect sensor positioning",
        "Quick obsolescence", "Frequent system updates needed", "Disruptive system features",
        "Bulky hardware", "Need for extensive software modifications", "Hardware/software compatibility issues",
        "Device visualization problems", "Equipment setup related difficulties",
        "Use of proprietary software creating obstacles in urgent clinical situations",
        "Technical challenges", "Technical difficulties during image scanning"
    ],

    # 2. Connectivity / Network / Internet
    "connectivity": [
        "connectivity problems", "network issues", "internet reliability issues", "limited bandwidth",
        "poor signal coverage", "wifi/cellular problems", "rural connectivity problems", "slow internet",
        "service interruptions", "unstable internet", "connection difficulties", "network outages",
        "internet lag", "poor network performance", "dropped connections", "network failures",
        "Limited connectivity", "Limited network reliability", "Slow Internet speed",
        "Poor audio quality", "Poor video quality", "Internet access issue", "Wireless issue or poor signal",
        "Communication technology problems (Bluetooth, Wi-Fi, cellular)",
        "Problems with antenna positioning", "Bandwidth and signal limitations", "Unreliable/expensive internet access",
        "Limited nationwide internet availability", "Telecom service issues",
        "Mobile internet connections being weaker than fixed networks (NBN)",
        "Insufficient internet bandwidth for stable connections", "Variable quality of videoconferencing tools"
    ],

    # 3. Usability / Interface / Design
    "usability_design": [
        "usability issues", "poor user interface", "complex navigation", "lack of user-friendliness",
        "interface design problems", "poorly designed interface", "complex or unclear UI",
        "difficult to use the system", "lack of customization", "limited personalization options",
        "overload/complicated information", "unengaging content", "generic designs",
        "inadequate software design", "high cognitive load", "small interactive elements",
        "lack of screen reader/voice assistance", "confusing layout", "inaccessible interface",
        "non-intuitive design", "poor user experience", "Limited tactile feedback capabilities",
        "Complex registration processes", "Poor interface readability", "inconvenient device placement",
        "Content not engaging or relevant", "A lack of interactive design features",
        "Technology design and usability", "Complex and unclear navigation structures with high cognitive load",
        "Complex technical terminology and ambiguous messaging", "Insufficient font sizes or contrast",
        "Display issues", "Need for single-screen layouts", "Lack of screen reader and voice assistance options",
        "Issues related to the interface and design of the remote platform", "inflexible format",
        "Poor usability of telehealth systems", "generic designs, complex user interface",
        "Difficult-to-use hardware and software", "Design flaws"
    ],

    # 4. Privacy / Security / Confidentiality
    "privacy_security": [
        "privacy concerns", "security concerns", "confidentiality issues", "data protection problems",
        "ethical constraints", "data misuse concerns", "lack of secure systems",
        "legal liability concerns related to data", "data privacy issues", "data breaches",
        "unauthorized access", "cybersecurity vulnerabilities", "security loopholes",
        "privacy risks", "information security issues", "inadequate encryption", "insecure data storage",
        "Privacy concerns or issues", "data safety concerns", "data security risks",
        "Anonymity", "data protection",
        "Lack of authentication centers and digital signatures",
        "Privacy violations through unsecured internet connections",
        "Difficulty in determining the user's location in cases of self-harm",
        "Concerns about privacy protection and conversation recording", "Concerns about data confidentiality and security",
        "Lack of anonymous reporting options", "confidentiality"
    ],

    # 5. Cost / Financial / Funding
    "cost_financial": [
        "cost prohibitive", "financial restrictions", "high equipment costs", "lack of funding",
        "reimbursement limitations", "insurance coverage problems", "limited affordability",
        "economic barriers", "implementation costs", "budget constraints", "financial burden",
        "lack of financial incentives", "high operational costs", "high maintenance expenses",
        "budget overruns", "expensive resources", "financial constraints", "funding challenges",
        "High startup costs", "High initial investment", "High program costs",
        "Limited insurance coverage",  # also appears in reimbursement
        "Hardware and software equipment costs", "Cellular bundle costs",
        "Innovation investment and financial risk", "High labor maintenance costs",
        "Initial equipment investment costs", "Slow confirmation of cost-benefit data",
        "Variable ongoing costs", "Personal financial burden", "Unclear economic costs",
        "Limited funding for development (e.g., rural specific software)", "expensive equipment requirements",
        "High costs"
    ],

    # 6. Training / Knowledge / Skills / Literacy
    "training_knowledge": [
        "lack of training", "limited technical literacy", "limited computer literacy",
        "lack of knowledge and education", "digital skills gaps", "low digital literacy",
        "lack of telemedicine training", "insufficient user instruction", "staff training shortages",
        "need for qualified instruction", "limited professional expertise", "knowledge deficits",
        "lack of expertise", "inadequate training", "skill gaps", "educational barriers",
        "insufficient learning resources", "deficient instructional support",
        "Low formal education", "Need for technical training", "Limited user training",
        "Low mental health literacy", "Limited ATT knowledge",
        "Aversion to/difficulty learning how to operate new technology", "Lack of confidence and technical skills",
        "computer anxiety", "Need for provider training (tech and emotional)",
        "Limited telehealth knowledge", "Lack of awareness of telemedicine/telerehabilitation",
        "Limited digital literacy among patients", "Low technological literacy among patients",
        "Professional skill degradation", "Content comprehension barriers",
        "Deters residents from learning evidence-based guidelines", "Lack of healthcare provider training",
        "Limited exposure/knowledge of eHealth", "Lack of support for usability for people with low computing skills",
        "computer illiteracy and lack of possessing telehealth equipment", "Lack of support for usability for patients with low computer skills",
        "Limited knowledge and skills among health workers", "Lack of local IT skilled work force in rural practices",
        "Lack of training and support", "Difficulty transferring records between systems",
        "Doubts about patients' ability to use technology", "Low health/digital literacy"
    ],

    # 7. Support / Assistance / Organizational Backing
    "support_assistance": [
        "technical support limitations", "lack of support from management", "no caregiver support",
        "insufficient workforce for monitoring", "shortage of technical support staff", "limited provider support",
        "staff shortages and turnover", "lack of organizational support", "insufficient administrative support",
        "deficient helpdesk resources", "poor service support", "lack of customer service", "inadequate assistance",
        "support deficits", "understaffed support teams", "Limited technical support", "Staff support limitations",
        "Need for assistance personnel (nurses or family members) to help with examinations",
        "Governance and management", "Lack of strategic implementation plan"
    ],

    # 8. Integration / Interoperability / Workflow
    "integration_workflow": [
        "system integration difficulties", "lack of integration into practice", "workflow disruption",
        "interoperability issues", "incompatibility with health record systems",
        "parallel systems in workflows", "difficulty merging with existing workflows",
        "need for specialized equipment/software", "integration challenges", "seamless integration issues",
        "system compatibility problems", "workflow misalignment", "lack of standard interfaces",
        "data silo issues", "integration gaps", "inefficient workflow integration",
        "Service integration challenges", "Low integration into provider work flow",
        "Lack of system cross-synchronization", "Issues with centralized scheduling/data systems",
        "No link between medical softwares", "Difficulty transferring records between systems",
        "Multiple parallel systems", "Varied operational requirements", "Provider workflow issues",
        "Difficulty in solving interoperability issue", "Inability to integrate into clinical workflow",
        "Multiple devices complicates monitoring", "Double documentation"
    ],

    # 9. Access / Availability / Coverage
    "access_availability": [
        "limited service locations", "limited platform accessibility", "device access limitations",
        "lack of necessary equipment", "limited coverage range", "lack of network coverage",
        "lack of computers/devices", "lack of broadband access", "limited telemedicine network coverage",
        "inaccessible functional videoconferencing", "resource constraints", "access restrictions",
        "geographical access issues", "unavailable services", "barriers to access", "limited reach",
        "Limited internet access", "Limited hardware access", "Limited device access",
        "Access inequalities", "Not having the required equipment", "Homelessness",
        "Limited access to computers", "Must own text-enabled cell phone", "Must own smart cell phone",
        "Platform access limited to computers", "Limited availability of internet access",
        "Limited service hours/availability", "Transportation inconvenience for accessing telemedicine healthcare",
        "Limited public transportation", "Availability of alternatives to e-Health services",
        "Limited program exposure", "Equipment accessibility", "Limited device availability"
    ],

    # 10. Quality / Performance / Reliability
    "quality_performance": [
        "limited system quality", "low perceived system effectiveness", "poor system performance",
        "inaccurate or unreliable data", "limited accuracy of clinical decision support systems",
        "unreliable system performance", "low reliability", "limited diagnostic capabilities",
        "lack of thoroughness vs. in-person care", "system inefficiency", "performance issues",
        "inconsistency in results", "subpar quality", "deficient performance", "variability in output",
        "Limited system performance", "Limited system quality", "Poor system performance",
        "Limited diagnostic accuracy", "Lack of accuracy or reliability", "Slow system response times",
        "Perceptions of effectiveness", "Unmet expectations about the system's capabilities",
        "Limited real-time feedback", "Inability to provide real-time access", "Reliability concerns",
        "Unplanned downtime"
    ],

    # 11. Psychosocial & Emotional Barriers / Stigma / Attitudes
    "psychosocial_emotional": [
        "patients prefer face-to-face", "resistance among healthcare professionals",
        "fear of using new technology", "stigma about ageing or mental illness",
        "negative attitudes or skepticism", "lack of trust", "mistrust and worry",
        "social stigma", "fear of judgement", "prejudices on telehealth", "bias",
        "fear of not understanding technology", "negative experience with technology",
        "fear or concerns about technology", "reluctance to change", "psychological resistance",
        "apprehension towards digital solutions", "emotional barriers", "concerns about impersonality",
        "Patients prefer in-person visits", "Resistance to technology", "Skepticism about digital care",
        "Psychological stress", "Emotional distress", "Anxiety about technology", "Fear of using technology",
        "Hard to express emotion (patients)", "Lack of body language", "Low physician communication skills",
        "Judging oneself or being judged negatively by physicians",
        "Medical data reminded patients of the negative aspects of their illness",
        "Fear of negative side effects", "Negative beliefs about help-seeking",
        "Patient anxiety", "Preference towards traditional medical practices", "Stigma reducing acceptance",
        "Isolation negatively impacting home telepsychiatry", "Trust issues towards telemedicine products",
        "express emotions"
    ],

    # 12. Time / Scheduling / Workload
    "time_workload": [
        "limited time resources", "increased workload for clinicians", "time constraints affecting sessions",
        "labor- and time-intensive for providers", "time-consuming processes", "staff resource limitations",
        "additional time requirements for setup/troubleshooting", "installation delays", "scheduling challenges",
        "time shortage", "excessive time demands", "time inefficiency", "delays in service", "long waiting times",
        "prolonged procedures", "Scheduling", "Time availability constraints",
        "After-hours care management issues", "Extra time required for system use",
        "Reduced working memory and focus capacity", "Daily reminders causing hypervigilance",
        "Virtual visits require extra time", "Provider fatigue and dissatisfaction",
        "Increases nursing activity", "Added workload related to telemedicine operations",
        "Lack of time due to work overload"
    ],

    # 13. Legal / Regulatory / Policy
    "legal_regulatory": [
        "legal framework gaps", "policy challenges", "complex medical regulations",
        "insufficient regulations", "licensing/regulation challenges", "lack of laws",
        "lack of telemedicine policy", "reimbursement and billing issues", "interstate licensure issues",
        "lack of standardized guidelines", "inconsistent government policy", "malpractice and legal risks",
        "no regulations available", "regulatory barriers", "policy inconsistencies", "compliance challenges",
        "legal obstacles", "statutory limitations", "Complex compliance requirements",
        "Unclear legal framework", "Legal concerns", "Regulatory uncertainties", "policy",
        "regulation and reform", "mental healthcare integration", "public and private mental healthcare systems",
        "Universal coverage", "Need for clear telehealth policies", "Non-standardized consent processes",
        "Different regulations between countries", "Unclear legal responsibility for immediate response",
        "Complex monitoring protocols", "Inadequate legal frameworks and regulations",
        "Medico-legal issues and liability", "Licensing regulations across states",
        "Complex regulatory requirements and varying state laws", "Lack of comprehensive telemedicine policy",
        "Inadequate governance, policies, and support structures", "Unclear regulation",
        "Modifying existing policy", "Legal and ethical considerations", "Absence or inadequacy of legislation, policies",
        "Interstate licensing challenges", "Pressure to follow protocol strictly"
    ],

    # 14. Infrastructure (Power, Facilities, ICT)
    "infrastructure": [
        "limited clinical infrastructure", "weak ICT infrastructure", "unreliable power supply",
        "infrastructure deficits in remote areas", "poor telecommunications structures", "inadequate telehealth infrastructure",
        "limited or outdated devices", "weather impacts on connectivity", "lack of integrated framework",
        "insufficient facilities", "inadequate physical infrastructure", "power outages", "obsolete systems",
        "deficient infrastructure", "inadequate network infrastructure", "limited facility resources",
        "Limited infrastructure", "Infrastructure challenges", "Poor IT infrastructure", "ICT infrastructure limitations",
        "Unreliable electricity supply", "Operational requirements for national implementation",
        "Poor equipment conditions in counseling centers", "Lack of electrical power", "Power supply issues"
    ],

    # 15. Scalability / Expansion / Growth
    "scalability": [
        "scalability issues", "scalability challenges", "limited scalability of services",
        "challenges in processing large datasets", "limited capacity to expand", "high program costs (scaling)",
        "insufficient infrastructure for scaling", "expansion difficulties", "growth limitations",
        "capacity constraints", "scaling barriers", "resource limitations for scaling", "limited system flexibility",
        "inadequate scalability", "Scalability challenges in processing large datasets"
    ],

    # 16. Clinical / Medical Limitations
    "clinical_limitations": [
        "limited clinical information vs. in-person visits", "inability to address certain conditions",
        "lack of physical examination", "telemedicine unsuited for emergencies", "virtual diagnoses risk misses",
        "clinical assessment is limited", "inadequate clinical detail", "lack of comprehensive care",
        "medical limitations", "inability to conduct hands-on exams", "diagnostic limitations",
        "Limited clinical assessment", "Risk of misdiagnosis",
        "comorbidities and worsening of the condition", "diabetes and heart disease",
        "Scepticism about the accuracy of the results", "Inability to completely replace in-person visits",
        "Constraints in specific specialties", "Inability to perform certain diagnostics remotely",
        "Exams less precise than in-person", "Replacement of existing therapy",
        "Inability to perform physical exams", "Non-updatable images due to opaque media like cataracts",
        "Limited ability to perform certain diagnostic tests (e.g., fluorescein staining)",
        "Risk of medical errors due to remote management", "Lack of information for diagnosis",
        "Increased medical liability",
        "Difficulty conducting medical evaluations virtually", "Concerns about care effectiveness",
        "Concerns about shared medical responsibility", "Not suitable for all patient types",
        "Inability to address certain skin conditions", "Potential selection bias",
        "Lack of a single diagnostic standard for non-melanocytic skin lesions (NMSC)",
        "Complexity of teledermatology procedures", "Risk of missing malignant lesions",
        "Lack of histopathological reports for all lesions", "Interrater disagreement in burn assessment",
        "Inability to provide accurate remote assessments", "Limited assessment tool accuracy",
        "Insufficient medical history data in teledermatology", "Concerns about clinical appropriateness, effectiveness, and safety",
        "Lack of solid evidence proving clinical benefits", "Undermines holistic surveillance",
        "Limited ability to perform physical exams", "Inadequate wound care training", "Inadequate clinical guidelines",
        "Difficulty in coordinating telehealth appointments", "Increased risk of misdiagnosis"
    ],

    # 17. Personal / Motivation / Engagement
    "personal_motivation": [
        "low perceived value or effectiveness", "lack of user engagement", "lack of interaction features",
        "patients not incorporated into design", "lack of motivation", "patient reluctance",
        "personal preferences", "aversion to new technology", "lack of routine",
        "disinterest of users", "lack of satisfaction", "feeling forced by others",
        "low commitment", "reduced user interest", "apathy", "lack of enthusiasm", "diminished engagement",
        "Patients' choice and preference", "Late involvement of Patients", "Use of ineffective engagement methods",
        "Perception of no practical benefits", "Difficulty engaging/maintaining use", "motivation"
    ],

    # 18. Language / Culture / Local Adaptation
    "language_culture": [
        "language barriers", "not culturally appropriate", "cultural adaptation challenges",
        "lack of cultural representation", "linguistic and cultural barriers", "cultural differences",
        "language differences", "local health beliefs", "insufficient cultural sensitivity",
        "language and culture issues", "communication barriers due to language",
        "inadequate cultural adaptation", "Need for multilingual text messaging",
        "Gender-related cultural practices", "Belief that teledermatology is less accurate than in-person consultation"
    ],

    # 19. Data Management / Standardization / Accuracy
    "data_management": [
        "data entry complexity", "lack of standardization", "data inaccuracy and misdiagnosis",
        "data accuracy concerns", "motion artifacts", "manual data upload",
        "lack of interoperability with data systems", "data management issues", "inconsistent data",
        "poor data quality", "data errors", "inaccurate data", "data inconsistencies", "recording errors",
        "data integration problems", "data synchronization issues", "Limited data standardization",
        "Poor data integration", "Data handling issues", "Limited document upload capability",
        "Manual data entry requirements", "Delays in real-time data analysis", "Low data utilization for planning",
        "Difficulty transmitting documents", "Challenges obtaining out-of-hospital records",
        "Content-related issues (broad context, relevancy, and complexity)"
    ],

    # 20. Environmental / Space / Physical Constraints & Sensory/Cognitive Impairments
    "environment_space": [
        "limited private space availability", "limited space at home for system", "lack of quiet environment",
        "physical limitation barriers", "environment constraints", "device size and weight", "space limitations",
        "physical access challenges", "inadequate spatial resources", "restricted physical environment",
        "limited room", "insufficient space", "Limited physical space", "Space constraints",
        "Limited vision ability", "cognitive impairments", "Old age", "Limited visual acuity",
        "Limited hand-eye coordination", "Limited motion estimation and peripheral vision",
        "Physical disabilities (e.g., poor eyesight)", "Physical ability"
    ],

    # 21. Organizational / Management Issues & Human Resources
    "organizational_issues": [
        "management support limitations", "bureaucratic delays", "recruitment challenges",
        "lack of incentives for health professionals", "lack of desire to change clinical paradigms",
        "poor community integration", "multiple non-compatible e-Health projects", "stakeholder conflicts",
        "administrative complexities", "organizational inefficiencies", "poor management", "internal conflicts",
        "lack of leadership", "management obstacles", "organizational disarray",
        "Human resource shortages", "Limited specialized developers", "Few practitioners available",
        "Developer-user disconnection", "Governance and management", "Complex implementation requirements",
        "Hierarchical structures affecting implementation", "Staff workload issues", "High turnover of medical and nursing staff",
        "Limited human resources", "Limited qualified personnel", "Limited training and support",
        "Lack of strategic implementation plan", "System sustainability issues",
        "Shortage of specialists", "Insufficient quantity of health and IT workers",
        "Limited equipment management ability", "Missing fit into organizational structures",
        "Poor national coordination", "Unmet expectations regarding coordination", "Lack of staff allocation"
    ],

    # 22. Reimbursement / Insurance / Financial Incentives
    "reimbursement_insurance": [
        "reimbursement issues", "lack of program reimbursement by insurance", "need for new reimbursement models",
        "different reimbursement schemes", "insurance/reimbursement inconsistencies", "limited reimbursement structures",
        "lack of established reimbursement models", "financial incentive issues", "billing problems",
        "insurance limitations", "complicated billing", "reimbursement challenges", "Payment system inadequacy",
        "Limited insurance coverage", "Lack of reimbursement structure", "Lack of provider compensation"
    ],

    # 23. Provider-Patient Relationship / Communication
    "provider_patient_communication": [
        "loss of personal interaction", "lack of emotional support", "deterioration of relationship with providers",
        "doctor-patient relationship impact", "lack of direct communication cues", "trust and rapport issues",
        "difficulties in maintaining dialogue", "communication breakdown", "impersonal communication",
        "strained relationships", "lack of interpersonal connection", "reduced empathy", "impaired rapport",
        "Poor provider-patient communication", "Limited interaction cues", "Timing of patient-provider interactions",
        "Lack of body language", "Low physician communication skills", "Difficulty understanding patient needs",
        "Lack of connection with therapist", "No physical presence to interpret body gestures/facial expressions",
        "Disruptions to conversation flow", "Concerns about shared medical responsibility",
        "Negative impact on doctor-patient relationship", "Loss of clinical autonomy",
        "Difficulty giving instructions and monitoring progress", "Communication challenges between doctors and patients",
        "Poor interpersonal rapport, empathy, and clinical correlation", "Concerns about clinician independence"
    ]
}




########################################################################
## 2. Text Cleaning & Preprocessing with Fuzzy Standardization
########################################################################

def fuzzy_standardize(text, term_dict, threshold=70):
    """
    Applies fuzzy matching on token n-grams to catch near-matches
    for phrases in term_dict. If an n-gram scores above the threshold
    (using RapidFuzz's ratio), it is replaced with the standardized key.
    """
    tokens = text.split()
    new_tokens = []
    i = 0
    # Process tokens sequentially
    while i < len(tokens):
        replaced = False
        # For each standard term, try each phrase
        for std_term, phrases in term_dict.items():
            for phrase in phrases:
                phrase_words = phrase.split()
                n = len(phrase_words)
                if i + n <= len(tokens):
                    candidate = " ".join(tokens[i:i+n])
                    score = fuzz.ratio(candidate, phrase)
                    if score >= threshold:
                        new_tokens.append(std_term)
                        i += n
                        replaced = True
                        break
            if replaced:
                break
        if not replaced:
            new_tokens.append(tokens[i])
            i += 1
    return " ".join(new_tokens)

def clean_text(text):
    """
    Cleans text by:
      1. Lower-casing
      2. Removing parentheticals and bullet-like prefixes
      3. Standardizing domain-specific synonyms using regex and fuzzy matching
      4. Removing extra punctuation
      5. Lemmatizing using spaCy
      6. Removing stopwords (both default and extended)
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Remove bullet prefixes or leading numbers
    text = re.sub(r'^\s*\d+[\).-]?\s*', '', text)
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'e\.g\.|i\.e\.', '', text)
    # Exact standardization using regex
    for std_term, phrases in TERM_STANDARDIZATION.items():
        pattern = r'\b(?:' + '|'.join(map(re.escape, phrases)) + r')\b'
        text = re.sub(pattern, std_term, text)
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Fuzzy matching to catch near-misses
    text = fuzzy_standardize(text, TERM_STANDARDIZATION, threshold=70)
    # Lemmatize and remove stopwords
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        lemma = token.lemma_
        if lemma == "-PRON-":
            lemma = token.text
        if token.is_stop or lemma in EXTENDED_STOPWORDS:
            continue
        tokens.append(lemma)
    return " ".join(tokens)

########################################################################
## 3. Universal Sentence Encoder, Clustering & Post-Processing
########################################################################

def load_use_model():
    print("Loading Universal Sentence Encoder model...")
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return model

def embed_texts_use(text_list, use_model):
    embeddings = use_model(text_list)
    embeddings_np = embeddings.numpy()
    embeddings_normalized = normalize(embeddings_np)
    return embeddings_normalized

def assign_cluster_names_tfidf(df, cluster_col="Cluster", cleaned_text_col="cleaned_text", top_n=3):
    """
    Uses TF‑IDF to extract representative terms from each cluster.
    The noise cluster (-1) is labeled as "Noise".
    """
    cluster_names = {}
    unique_clusters = sorted(df[cluster_col].unique())
    for clus_id in unique_clusters:
        if clus_id == -1:
            cluster_names[clus_id] = "Noise"
            continue
        texts = df[df[cluster_col] == clus_id][cleaned_text_col].tolist()
        if not texts:
            cluster_names[clus_id] = f"Topic{clus_id}: misc"
            continue
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            cluster_names[clus_id] = f"Topic{clus_id}: misc"
            continue
        tfidf_means = tfidf_matrix.mean(axis=0).A1
        terms = vectorizer.get_feature_names_out()
        if len(tfidf_means) == 0:
            cluster_names[clus_id] = f"Topic{clus_id}: misc"
            continue
        top_indices = tfidf_means.argsort()[-top_n:][::-1]
        top_terms = [terms[i] for i in top_indices]
        cluster_names[clus_id] = f"Topic{clus_id}: " + " / ".join(top_terms)
    df["ClusterName"] = df["Cluster"].map(cluster_names)
    return df

def merge_similar_clusters(df, embeddings, similarity_threshold=0.9):
    """
    Post-processes clustering results by computing centroids for each non-noise cluster
    and merging clusters whose centroids have cosine similarity above the threshold.
    Uses a union-find method to merge clusters transitively.
    """
    unique_clusters = sorted([c for c in df["Cluster"].unique() if c != -1])
    if not unique_clusters:
        return df  # Nothing to merge if only noise exists
    centroids = {}
    for clus in unique_clusters:
        indices = df.index[df["Cluster"] == clus].tolist()
        centroids[clus] = np.mean(embeddings[indices], axis=0)
    merge_pairs = []
    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            c1 = unique_clusters[i]
            c2 = unique_clusters[j]
            sim = cosine_similarity(centroids[c1].reshape(1, -1), centroids[c2].reshape(1, -1))[0, 0]
            if sim >= similarity_threshold:
                merge_pairs.append((c1, c2))
    # Union-Find structure for merging clusters
    parent = {cid: cid for cid in unique_clusters}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    def union(x, y):
        rootX = find(x)
        rootY = find(y)
        if rootX != rootY:
            parent[rootY] = rootX
    for (c1, c2) in merge_pairs:
        union(c1, c2)
    new_labels = {cid: find(cid) for cid in unique_clusters}
    # Update the df cluster labels; noise (-1) remains unchanged
    df["Cluster"] = df["Cluster"].apply(lambda x: new_labels[x] if x in new_labels else x)
    return df

def cluster_texts_hdbscan(df, text_col, use_model, min_cluster_size=5, min_samples=None):
    # Reset index to ensure contiguous indexing for proper alignment with embeddings
    df = df.reset_index(drop=True)
    df["cleaned_text"] = df[text_col].apply(clean_text)
    texts = df["cleaned_text"].tolist()
    embeddings = embed_texts_use(texts, use_model)
    n_samples, n_features = embeddings.shape
    n_components = min(n_features, n_samples, 50)  # safeguard for PCA
    pca = PCA(n_components=n_components, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric='euclidean')
    labels = clusterer.fit_predict(embeddings_pca)
    df["Cluster"] = labels
    # Post-process: merge similar clusters (noise remains as -1)
    df = merge_similar_clusters(df, embeddings_pca, similarity_threshold=0.9)
    df = assign_cluster_names_tfidf(df, cluster_col="Cluster", cleaned_text_col="cleaned_text")
    return df, clusterer, embeddings_pca


def calculate_intra_cluster_similarity(embeddings, labels):
    results = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_emb = embeddings[cluster_indices]
        if len(cluster_emb) > 1:
            sim_matrix = cosine_similarity(cluster_emb)
            np.fill_diagonal(sim_matrix, 0)
            n = len(cluster_emb)
            sum_upper = np.triu(sim_matrix, 1).sum()
            denom = (n * (n - 1)) / 2
            avg_sim = sum_upper / denom if denom > 0 else 0
        else:
            avg_sim = 1.0
        results[label] = avg_sim
    return results

def print_cluster_summary(df, embeddings_pca):
    labels = df["Cluster"].values
    intra_sim = calculate_intra_cluster_similarity(embeddings_pca, labels)
    cluster_info = []
    unique_labels = sorted(df["Cluster"].unique())
    for label in unique_labels:
        count = (labels == label).sum()
        cluster_name = df[df["Cluster"] == label]["ClusterName"].iloc[0]
        sim = intra_sim[label] if label in intra_sim else None
        cluster_info.append({
            "Cluster": label,
            "ClusterName": cluster_name,
            "Count": count,
            "IntraClusterSimilarity": sim
        })
    summary_df = pd.DataFrame(cluster_info)
    return summary_df

def recluster_noise(df, text_col, use_model, min_cluster_size=3, min_samples=None):
    """
    Extracts noise points (Cluster == -1) and re-clusters them using HDBSCAN.
    """
    noise_df = df[df["Cluster"] == -1].copy()
    if noise_df.empty:
        return None, None, None
    print("Re-clustering noise points...")
    new_df, new_clusterer, new_embeddings = cluster_texts_hdbscan(noise_df, text_col, use_model,
                                                                  min_cluster_size=min_cluster_size,
                                                                  min_samples=min_samples)
    return new_df, new_clusterer, new_embeddings


########################################################################
## 5. Reading & Splitting Excel (Including Income_level)
########################################################################

def read_and_split_excel(excel_file_path: str):
    df = pd.read_excel(excel_file_path, engine="openpyxl")
    required_cols = ["Author", "Year", "Barriers", "Facilitators", "Income_level"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in Excel file.")
    barrier_rows = []
    for idx, row in df.iterrows():
        author = row["Author"]
        year = row["Year"]
        income = row["Income_level"]
        barriers = row["Barriers"]
        if pd.isna(barriers) or str(barriers).strip().upper() == "N/A":
            continue
        for b in str(barriers).split("\n"):
            b = b.strip()
            if b:
                barrier_rows.append((author, year, income, b))
    barriers_df = pd.DataFrame(barrier_rows, columns=["Author", "Year", "Income_level", "Barrier"])

    facilitator_rows = []
    for idx, row in df.iterrows():
        author = row["Author"]
        year = row["Year"]
        income = row["Income_level"]
        facilitators = row["Facilitators"]
        if pd.isna(facilitators) or str(facilitators).strip().upper() == "N/A":
            continue
        for f in str(facilitators).split("\n"):
            f = f.strip()
            if f:
                facilitator_rows.append((author, year, income, f))
    facilitators_df = pd.DataFrame(facilitator_rows, columns=["Author", "Year", "Income_level", "Facilitator"])
    return barriers_df, facilitators_df

########################################################################
## 6. Main Script
########################################################################

def main():
    excel_file = "/Users/angelocapodici/Desktop/data.xlsx"
    use_model = load_use_model()
    barriers_df, facilitators_df = read_and_split_excel(excel_file)

    # --- Overall Clustering for Barriers ---
    print("=== Clustering Barriers (HDBSCAN) ===")
    barriers_df, barrier_clusterer, barrier_pca = cluster_texts_hdbscan(
        barriers_df,
        text_col="Barrier",
        use_model=use_model,
        min_cluster_size=5
    )
    barrier_summary = print_cluster_summary(barriers_df, barrier_pca)
    print("\nOverall Barriers Cluster Summary (after merging similar clusters):")
    print(barrier_summary)
    barriers_df[["Author", "Year", "Income_level", "Barrier", "Cluster", "ClusterName"]].to_csv("barriers_output.csv", index=False)
    barrier_summary.to_csv("barriers_cluster_summary.csv", index=False)
    # plot_cluster_counts(barrier_summary, title="Overall Barriers: Count per Cluster")
    # plot_clusters_scatter(barrier_pca, barriers_df["Cluster"].values, title="Overall Barriers: Cluster Scatter Plot")

    # Re-cluster noise for overall barriers
    if (barriers_df["Cluster"] == -1).any():
        noise_barriers_df, noise_barrier_clusterer, noise_barrier_pca = recluster_noise(
            barriers_df, "Barrier", use_model, min_cluster_size=3
        )
        if noise_barriers_df is not None:
            noise_barrier_summary = print_cluster_summary(noise_barriers_df, noise_barrier_pca)
            print("\nRe-clustered Noise (Barriers) Summary:")
            print(noise_barrier_summary)
            noise_barrier_summary.to_csv("barriers_noise_cluster_summary.csv", index=False)
            noise_barriers_df[["Author", "Year", "Income_level", "Barrier", "Cluster", "ClusterName"]].to_csv("barriers_noise_output.csv", index=False)
            # plot_cluster_counts(noise_barrier_summary, title="Noise Barriers: Count per Cluster")
            # plot_clusters_scatter(noise_barrier_pca, noise_barriers_df["Cluster"].values, title="Noise Barriers: Cluster Scatter Plot")

    # --- Overall Clustering for Facilitators ---
    print("\n=== Clustering Facilitators (HDBSCAN) ===")
    facilitators_df, facilitator_clusterer, facilitators_pca = cluster_texts_hdbscan(
        facilitators_df,
        text_col="Facilitator",
        use_model=use_model,
        min_cluster_size=5
    )
    facilitator_summary = print_cluster_summary(facilitators_df, facilitators_pca)
    print("\nOverall Facilitators Cluster Summary (after merging similar clusters):")
    print(facilitator_summary)
    facilitators_df[["Author", "Year", "Income_level", "Facilitator", "Cluster", "ClusterName"]].to_csv("facilitators_output.csv", index=False)
    facilitator_summary.to_csv("facilitators_cluster_summary.csv", index=False)
    # plot_cluster_counts(facilitator_summary, title="Overall Facilitators: Count per Cluster")
    # plot_clusters_scatter(facilitators_pca, facilitators_df["Cluster"].values, title="Overall Facilitators: Cluster Scatter Plot")

    # Re-cluster noise for overall facilitators
    if (facilitators_df["Cluster"] == -1).any():
        noise_facilitators_df, noise_facilitator_clusterer, noise_facilitators_pca = recluster_noise(
            facilitators_df, "Facilitator", use_model, min_cluster_size=3
        )
        if noise_facilitators_df is not None:
            noise_facilitator_summary = print_cluster_summary(noise_facilitators_df, noise_facilitators_pca)
            print("\nRe-clustered Noise (Facilitators) Summary:")
            print(noise_facilitator_summary)
            noise_facilitator_summary.to_csv("facilitators_noise_cluster_summary.csv", index=False)
            noise_facilitators_df[["Author", "Year", "Income_level", "Facilitator", "Cluster", "ClusterName"]].to_csv("facilitators_noise_output.csv", index=False)
            # plot_cluster_counts(noise_facilitator_summary, title="Noise Facilitators: Count per Cluster")
            # plot_clusters_scatter(noise_facilitators_pca, noise_facilitators_df["Cluster"].values, title="Noise Facilitators: Cluster Scatter Plot")

    # --- Clustering by Income Level for Barriers ---
    income_levels = sorted(barriers_df["Income_level"].unique())
    for level in income_levels:
        print(f"\n=== Clustering Barriers for Income Level: {level} ===")
        group_barriers = barriers_df[barriers_df["Income_level"] == level].copy()
        if group_barriers.empty:
            continue
        group_barriers, group_barrier_clusterer, group_barrier_pca = cluster_texts_hdbscan(
            group_barriers, text_col="Barrier", use_model=use_model, min_cluster_size=5
        )
        group_barrier_summary = print_cluster_summary(group_barriers, group_barrier_pca)
        print(f"\nBarriers Cluster Summary for Income Level {level}:")
        print(group_barrier_summary)
        group_barrier_summary.to_csv(f"barriers_cluster_summary_{level}.csv", index=False)
        group_barriers[["Author", "Year", "Income_level", "Barrier", "Cluster", "ClusterName"]].to_csv(f"barriers_output_{level}.csv", index=False)
        # plot_cluster_counts(group_barrier_summary, title=f"Barriers ({level}): Count per Cluster")
        # plot_clusters_scatter(group_barrier_pca, group_barriers["Cluster"].values, title=f"Barriers ({level}): Cluster Scatter Plot")
        if (group_barriers["Cluster"] == -1).any():
            noise_group_barriers, noise_group_barrier_clusterer, noise_group_barrier_pca = recluster_noise(
                group_barriers, "Barrier", use_model, min_cluster_size=3
            )
            if noise_group_barriers is not None:
                noise_group_barrier_summary = print_cluster_summary(noise_group_barriers, noise_group_barrier_pca)
                print(f"\nRe-clustered Noise (Barriers) Summary for Income Level {level}:")
                print(noise_group_barrier_summary)
                noise_group_barrier_summary.to_csv(f"barriers_noise_cluster_summary_{level}.csv", index=False)
                noise_group_barriers[["Author", "Year", "Income_level", "Barrier", "Cluster", "ClusterName"]].to_csv(f"barriers_noise_output_{level}.csv", index=False)
                # plot_cluster_counts(noise_group_barrier_summary, title=f"Noise Barriers ({level}): Count per Cluster")
                # plot_clusters_scatter(noise_group_barrier_pca, noise_group_barriers["Cluster"].values, title=f"Noise Barriers ({level}): Cluster Scatter Plot")

    # --- Clustering by Income Level for Facilitators ---
    income_levels = sorted(facilitators_df["Income_level"].unique())
    for level in income_levels:
        print(f"\n=== Clustering Facilitators for Income Level: {level} ===")
        group_facilitators = facilitators_df[facilitators_df["Income_level"] == level].copy()
        if group_facilitators.empty:
            continue
        group_facilitators, group_facilitator_clusterer, group_facilitator_pca = cluster_texts_hdbscan(
            group_facilitators, text_col="Facilitator", use_model=use_model, min_cluster_size=5
        )
        group_facilitator_summary = print_cluster_summary(group_facilitators, group_facilitator_pca)
        print(f"\nFacilitators Cluster Summary for Income Level {level}:")
        print(group_facilitator_summary)
        group_facilitator_summary.to_csv(f"facilitators_cluster_summary_{level}.csv", index=False)
        group_facilitators[["Author", "Year", "Income_level", "Facilitator", "Cluster", "ClusterName"]].to_csv(f"facilitators_output_{level}.csv", index=False)
        # plot_cluster_counts(group_facilitator_summary, title=f"Facilitators ({level}): Count per Cluster")
        # plot_clusters_scatter(group_facilitator_pca, group_facilitators["Cluster"].values, title=f"Facilitators ({level}): Cluster Scatter Plot")
        if (group_facilitators["Cluster"] == -1).any():
            noise_group_facilitators, noise_group_facilitator_clusterer, noise_group_facilitator_pca = recluster_noise(
                group_facilitators, "Facilitator", use_model, min_cluster_size=3
            )
            if noise_group_facilitators is not None:
                noise_group_facilitator_summary = print_cluster_summary(noise_group_facilitators, noise_group_facilitator_pca)
                print(f"\nRe-clustered Noise (Facilitators) Summary for Income Level {level}:")
                print(noise_group_facilitator_summary)
                noise_group_facilitator_summary.to_csv(f"facilitators_noise_cluster_summary_{level}.csv", index=False)
                noise_group_facilitators[["Author", "Year", "Income_level", "Facilitator", "Cluster", "ClusterName"]].to_csv(f"facilitators_noise_output_{level}.csv", index=False)
                # plot_cluster_counts(noise_group_facilitator_summary, title=f"Noise Facilitators ({level}): Count per Cluster")
                # plot_clusters_scatter(noise_group_facilitator_pca, noise_group_facilitators["Cluster"].values, title=f"Noise Facilitators ({level}): Cluster Scatter Plot")

if __name__ == "__main__":
    main()
