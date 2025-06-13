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
    "same", "so", "than", "too", "very", "just", "can", "will",
    # Domain filler words
    "lack", "lack of", "limited", "issues", "issue", "barrier", "barriers", "concern"
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

