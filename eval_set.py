# Hand-labeled evaluation set
# Each entry: question, expected_source (doc filename), expected_answer_keywords (must appear in answer)

EVAL_SET = [
    # --- Original 10 ---
    {
        "id": "Q01",
        "question": "What is the minimum offset obligation percentage under DAP 2020?",
        "expected_source": "dap2020_sample.txt",
        "expected_keywords": ["30%", "30 percent", "thirty percent", "minimum of 30", "30 per cent"],
    },
    {
        "id": "Q02",
        "question": "How many Rafale aircraft did India order and what was the contract value?",
        "expected_source": "docs/major_acquisitions.txt",
        "expected_keywords": ["36", "59,000", "59000"],
    },
    {
        "id": "Q03",
        "question": "What is the indigenous content requirement for submarines under the Strategic Partner model?",
        "expected_source": "docs/strategic_partner.txt",
        "expected_keywords": ["45%", "45 percent"],
    },
    {
        "id": "Q04",
        "question": "What is DRDO's annual budget for 2023-24?",
        "expected_source": "docs/drdo.txt",
        "expected_keywords": ["23,000", "23000"],
    },
    {
        "id": "Q05",
        "question": "What FDI percentage is allowed under automatic route in Indian defence?",
        "expected_source": "docs/fdi_defence.txt",
        "expected_keywords": ["74%", "74 percent"],
    },
    {
        "id": "Q06",
        "question": "Who chairs the Defence Acquisition Council?",
        "expected_source": "docs/dac_decisions.txt",
        "expected_keywords": ["Defence Minister", "defense minister"],
    },
    {
        "id": "Q07",
        "question": "What is the Fast Track Procedure timeline for procurement?",
        "expected_source": "docs/dac_decisions.txt",
        "expected_keywords": ["6 months", "six months"],
    },
    {
        "id": "Q08",
        "question": "What is India's defence export target?",
        "expected_source": "docs/defence_budget.txt",
        "expected_keywords": ["35,000", "35000"],
    },
    {
        "id": "Q09",
        "question": "How many DRDO laboratories are there in India?",
        "expected_source": "docs/drdo.txt",
        "expected_keywords": ["52"],
    },
    {
        "id": "Q10",
        "question": "What are the two Defence Industrial Corridors and which states are they in?",
        "expected_source": "docs/fdi_defence.txt",
        "expected_keywords": ["Uttar Pradesh", "Tamil Nadu", "UP DIC", "TN DIC", "Lucknow", "Chennai", "Coimbatore"],
    },

    # --- New 15 (Q11–Q25) ---
    {
        "id": "Q11",
        "question": "What is India's total defence budget for 2023-24?",
        "expected_source": "docs/defence_budget.txt",
        "expected_keywords": ["5,93,537", "593537", "5.93 lakh"],
    },
    {
        "id": "Q12",
        "question": "What percentage of GDP does India's defence budget represent?",
        "expected_source": "docs/defence_budget.txt",
        "expected_keywords": ["1.98%", "1.98 percent"],
    },
    {
        "id": "Q13",
        "question": "What is the capital outlay for defence in 2023-24?",
        "expected_source": "docs/defence_budget.txt",
        "expected_keywords": ["1,62,600", "162600"],
    },
    {
        "id": "Q14",
        "question": "What is India's defence pension budget for 2023-24?",
        "expected_source": "docs/defence_budget.txt",
        "expected_keywords": ["1,38,205", "138205"],
    },
    {
        "id": "Q15",
        "question": "What is HAL's current order book value?",
        "expected_source": "docs/defence_budget.txt",
        "expected_keywords": ["94,000", "94000"],
    },
    {
        "id": "Q16",
        "question": "What was the cost of the Prachand Light Combat Helicopter order?",
        "expected_source": "docs/dac_decisions.txt",
        "expected_keywords": ["45,000", "45000"],
    },
    {
        "id": "Q17",
        "question": "What is Acceptance of Necessity in Indian defence procurement?",
        "expected_source": "docs/dac_decisions.txt",
        "expected_keywords": ["AoN", "Acceptance of Necessity", "first formal step", "Defence Acquisition Council"],
    },
    {
        "id": "Q18",
        "question": "What is the minimum indigenous content for Buy (Indian) category?",
        "expected_source": "docs/dac_decisions.txt",
        "expected_keywords": ["40%", "40 percent"],
    },
    {
        "id": "Q19",
        "question": "How many Apache helicopters did India procure and at what cost?",
        "expected_source": "docs/major_programmes.txt",
        "expected_keywords": ["22", "930"],
    },
    {
        "id": "Q20",
        "question": "When was INS Vikrant commissioned and what did it cost?",
        "expected_source": "docs/major_programmes.txt",
        "expected_keywords": ["2022", "20,000", "20000"],
    },
    {
        "id": "Q21",
        "question": "What is the range of the BrahMos missile?",
        "expected_source": "docs/major_programmes.txt",
        "expected_keywords": ["290 km", "450 km", "290 to 450", "290-450", "290km", "450km"],
    },
    {
        "id": "Q22",
        "question": "What was the value of India's BrahMos export deal with Philippines?",
        "expected_source": "docs/major_programmes.txt",
        "expected_keywords": ["375", "$375 million"],
    },
    {
        "id": "Q23",
        "question": "What is the range of the Agni-V ballistic missile?",
        "expected_source": "docs/major_programmes.txt",
        "expected_keywords": ["5,000 km", "5000 km"],
    },
    {
        "id": "Q24",
        "question": "How many Tejas Mk-1A jets did IAF order in total?",
        "expected_source": "docs/major_programmes.txt",
        "expected_keywords": ["180", "83", "97"],
    },
    {
        "id": "Q25",
        "question": "What is the indigenous content percentage of Tejas Mk-1?",
        "expected_source": "docs/major_programmes.txt",
        "expected_keywords": ["59.7%", "59.7 percent"],
    },
]
