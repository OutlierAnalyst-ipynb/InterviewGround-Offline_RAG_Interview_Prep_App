import os
import shutil
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configuration & Setup
# Make sure you have Ollama running with these models pulled!

EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "qwen2.5-coder:latest"
VECTOR_STORE_DIR = "./vector_store"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4

st.set_page_config(page_title="InterviewGround", layout="wide", page_icon="🎯")

# 2. Session State Management
# Keeping track of data so the page doesn't reset on every click

SESSION_DEFAULTS = {
    "vectorstore_built": False,
    "vectorstore": None,
    "questions": None,
    "answers": None,
    "retrieved_chunks": None,
    "jd_text": "",
}
for _key, _value in SESSION_DEFAULTS.items():
    st.session_state.setdefault(_key, _value)


# 3. Document Processing Pipeline
# Loading files and chopping them into manageable pieces

def load_document(uploaded_file):
    # Langchain likes file paths, so we temporarily save the uploaded file
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    if suffix not in (".pdf", ".txt"):
        raise ValueError("Only .pdf and .txt files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
        return loader.load()
    finally:
        os.remove(tmp_path)


def build_vector_store(resume_file, jd_file):
    # Parse the documents
    resume_docs = load_document(resume_file)
    jd_docs = load_document(jd_file)

    # Split the resume into chunks, respecting paragraphs and bullet points
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "•", "-", " "],
    )
    resume_chunks = splitter.split_documents(resume_docs)
    jd_text = "\n".join(doc.page_content for doc in jd_docs)

    # Note: We removed the shutil.rmtree() block completely!

    # Create an IN-MEMORY vector store (no persist_directory)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=resume_chunks,
        embedding=embeddings,
        # persist_directory=VECTOR_STORE_DIR # <-- Removed this line!
    )
    return vectorstore, jd_text

def retrieve_resume_context(vectorstore, jd_text, k=TOP_K):
    # Fetch the resume bullet points that best match the job description
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(jd_text)


# 4. LLM Prompts & Generation Chains
# The brains of the operation

QUESTION_PROMPT = PromptTemplate(
    input_variables=["jd", "context"],
    template="""You are an expert technical interviewer for Data Analytics,
Data Science, and AI/ML roles.

Your task is to analyze the Job Description (JD) against the Candidate Resume
Extracts and generate exactly 5 strong, candidate-specific interview questions.

IMPORTANT:
The Candidate Resume Extracts are the ONLY source of truth about the candidate.
Do not use outside knowledge to invent candidate experience.

CRITICAL PROVENANCE / ANTI-HALLUCINATION RULES:

1. NEVER mix, merge, or combine different resume experiences.

2. A Project, Internship, Job, Research Experience, Academic Experience, or
   Certification must remain associated ONLY with the exact heading/context
   where it appears.

3. NEVER assume that a project was performed during an internship or at a company
   unless the resume explicitly states that relationship.

4. For example:
   - If "Credit Risk Modeling" appears under "Projects", treat it ONLY as a project.
   - Do NOT write "During your internship at XYZ, you built the Credit Risk Model"
     unless the resume explicitly says that XYZ was where the project was done.

5. NEVER transfer:
   - technologies
   - datasets
   - metrics
   - responsibilities
   - achievements
   - business impact
   - tools
   - model names
   - results
   from one experience to another.

6. NEVER assume a metric belongs to a project unless the context explicitly
   places that metric in that project.

7. NEVER invent missing numbers, percentages, model performance, business impact,
   dataset size, duration, role, responsibility, or outcome.

8. If a fact is not explicitly stated in the resume context, DO NOT use that fact
   in the question.

9. Each question must be traceable to ONE specific resume experience.

10. If fewer than 5 distinct experiences exist, create multiple deeper questions
    about the explicitly available experiences instead of inventing new experiences.


QUESTION QUALITY RULES:

Generate questions that test the candidate's REAL experience against the JD.

Prioritize:
- technical implementation
- methodology
- model selection
- feature engineering
- data preprocessing
- SQL / Python / Excel / Power BI
- ML evaluation
- business reasoning
- problem solving
- trade-offs
- challenges
- validation
- deployment, ONLY if explicitly present
- measurable results, ONLY if explicitly present

Do NOT generate generic questions such as:
- "Tell me about your project."
- "What did you learn?"
- "Why did you choose this field?"

Instead, make questions specific to the actual resume content.

JD SKILL MATCHING:

Identify the important skills explicitly required by the JD.

Compare those skills ONLY against skills explicitly present in the resume context.

Separate them into:

MATCHED SKILLS:
Skills required by the JD and explicitly present in the resume.

MISSING / ADDITIONAL SKILLS:
Skills required by the JD but NOT explicitly present in the resume.

DO NOT assume that related skills are equivalent.

For example:
- Python does NOT automatically mean SQL.
- Machine Learning does NOT automatically mean Deep Learning.
- Pandas does NOT automatically mean Power BI.
- Statistics does NOT automatically mean A/B Testing.

If all relevant JD skills are explicitly present:
"Candidate has all required skills mentioned in the JD."

If the JD does not contain meaningful skill requirements:
"Skill set are not mentioned."


MANDATORY OUTPUT FORMAT:


Question 1:
<question>

Question 2:
<question>

Question 3:
<question>

Question 4:
<question>

Question 5:
<question>

### Skill Match

Matched Skills:
- <skill>
- <skill>

Missing / Additional Skills:
- <skill>
- <skill>

IMPORTANT:
- ALWAYS include Question 1 through Question 5.
- ALWAYS include the "### Skill Match" section.
- NEVER output "Question:" without its number.
- NEVER output "1." separately from the question.
- NEVER add introductions, explanations, reasoning, or commentary.

Job Description:
{jd}

Candidate Resume Extracts:
{context}
""",
)

ANSWER_PROMPT = PromptTemplate(
    input_variables=["jd", "context", "questions"],
    template="""You are an expert interview coach for Data Analytics,
Data Science, and AI/ML roles.

Your task is to generate detailed, interview-ready answers to the 5 questions
provided below.

The answers MUST be grounded strictly in the Candidate Resume Extracts.


CRITICAL ANTI-HALLUCINATION RULESL:


1. The resume context is the ONLY source of truth.

2. NEVER invent information.

3. NEVER combine information from different projects, internships, companies,
   jobs, research experiences, or academic experiences.

4. NEVER assume that an independent/academic project was completed during an
   internship or at a company.

5. Preserve the exact ownership of each experience.

6. If the resume says a project is a PROJECT, call it a project.

7. If the resume explicitly says something was done during an INTERNSHIP,
   attribute it to that internship.

8. If the resume does not explicitly state the relationship, DO NOT create one.

9. NEVER invent:
   - metrics
   - percentages
   - accuracy
   - ROC-AUC
   - precision
   - recall
   - F1 score
   - revenue
   - business impact
   - dataset size
   - number of users
   - duration
   - team size
   - deployment details
   - responsibilities
   - tools
   - technologies
   - achievements

10. If a requested detail is not present in the resume context, explicitly say:
   "The resume context does not specify this."

11. DO NOT convert one metric into another.

For example:
If the context says ROC-AUC = 0.936,
DO NOT write "accuracy = 93.6%".

ROC-AUC, accuracy, precision, recall, and F1 are different metrics.

12. DO NOT claim that a model "improved performance" unless the resume explicitly
    provides evidence of improvement.

13. Do not add technical steps that are not stated in the resume merely because
    they would be standard practice.

14. Keep each answer attached to the exact source experience used by the question.


ANSWER QUALITY:

The answers must NOT be tiny or one-paragraph generic answers.

For EACH question, provide a strong interview-ready answer of approximately
150–250 words when the available resume context supports that level of detail.

Use STAR where appropriate:

**Situation:**
Explain the actual situation from the resume.

**Task:**
Explain the actual responsibility/problem stated in the resume.

**Action:**
Explain what the candidate explicitly did.

**Result:**
Explain only the explicitly stated outcome/metric.

If one of these STAR components is NOT present in the resume,
DO NOT invent it.

Instead write:
"Not explicitly stated in the resume context."


ANSWER STYLE:

The answer should sound like a candidate speaking naturally in an interview.

It should:
- be specific
- be technically clear
- use first person ("I", "my")
- explain reasoning where supported
- mention tools/models actually present in the resume
- avoid unnecessary jargon
- avoid repeating the question
- avoid generic filler

Do NOT give an answer that merely paraphrases the question.

SOURCE CITATION:

At the end of every answer add:

**Source Citation:** <EXACT Project Title OR EXACT Company/Internship Name>

IMPORTANT:
- If the experience is a project, cite the exact project title.
- If the experience is an internship/company experience, cite the exact company
  or internship heading.
- NEVER cite a company for a project unless the context explicitly connects them.

SKILL MATCH:

After answering all 5 questions, add:

## Skill Match

**Matched Skills**
- Skills explicitly required by the JD and explicitly present in the resume.

**Missing / Additional Skills**
- Skills required by the JD but not explicitly present in the resume.

Do NOT infer equivalent skills.

If all required skills are present, write:

**Missing / Additional Skills**
Candidate has all required skills mentioned in the JD.

If the JD does not specify meaningful skills, write:

**Missing / Additional Skills**
Skill set are not mentioned.


MANDATORY OUTPUT FORMAT:

# Question 1
<full question>

### Model Answer

**Situation:** ...
**Task:** ...
**Action:** ...
**Result:** ...

**Source Citation:** ...

# Question 2
<full question>

### Model Answer

**Situation:** ...
**Task:** ...
**Action:** ...
**Result:** ...

**Source Citation:** ...

# Question 3
<full question>

### Model Answer

**Situation:** ...
**Task:** ...
**Action:** ...
**Result:** ...

**Source Citation:** ...

# Question 4
<full question>

### Model Answer

**Situation:** ...
**Task:** ...
**Action:** ...
**Result:** ...

**Source Citation:** ...

# Question 5
<full question>

### Model Answer

**Situation:** ...
**Task:** ...
**Action:** ...
**Result:** ...

**Source Citation:** ...

## Skill Match

**Matched Skills**
- ...

**Missing / Additional Skills**
- ...

DO NOT output any introduction, conclusion, commentary, or explanation outside
this format.

Job Description:
{jd}

Candidate Resume Extracts:
{context}

Interview Questions:
{questions}
""",
)

def generate_questions(jd_text, context_text):
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.7)
    chain = QUESTION_PROMPT | llm | StrOutputParser()
    return chain.invoke({"jd": jd_text, "context": context_text})


def generate_answers(jd_text, context_text, questions):
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.7)
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    return chain.invoke({"jd": jd_text, "context": context_text, "questions": questions})


# 5. Sidebar Layout & Controls
# Where the user uploads files and triggers the pipeline

with st.sidebar:
    st.header("⚙️ Configuration")
    st.caption(f"Embeddings: `{EMBED_MODEL}` · Chat: `{CHAT_MODEL}` (via Ollama)")

    resume_file = st.file_uploader("Resume", type=["pdf", "txt"])
    jd_file = st.file_uploader("Job Description", type=["pdf", "txt"])

    if st.button("1. Build Vector Store", type="primary", disabled=not (resume_file and jd_file)):
        try:
            with st.spinner("Loading and embedding resume..."):
                vectorstore, jd_text = build_vector_store(resume_file, jd_file)
                st.session_state.vectorstore = vectorstore
                st.session_state.jd_text = jd_text
                st.session_state.vectorstore_built = True

                # Reset downstream tasks since we have new documents
                st.session_state.questions = None
                st.session_state.answers = None
                st.session_state.retrieved_chunks = None
            st.success("Vector store built successfully!")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")

    if st.button("2. Generate Interview Q&A", disabled=not st.session_state.vectorstore_built):
        try:
            with st.spinner("Analyzing JD and retrieving resume context..."):
                retrieved_docs = retrieve_resume_context(
                    st.session_state.vectorstore, st.session_state.jd_text
                )
                st.session_state.retrieved_chunks = retrieved_docs
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

            with st.spinner("Generating targeted interview questions..."):
                st.session_state.questions = generate_questions(
                    st.session_state.jd_text, context_text
                )

            with st.spinner("Drafting grounded model answers & citations..."):
                st.session_state.answers = generate_answers(
                    st.session_state.jd_text, context_text, st.session_state.questions
                )
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.vectorstore_built:
        st.markdown("---")
        if st.button("↺ Start Over"):
            for key, value in SESSION_DEFAULTS.items():
                st.session_state[key] = value
            st.rerun()

# 6. Main UI Layout
# The central dashboard displaying results

st.title("🎯 InterviewGround")
st.markdown("Tailored interview prep, grounded in your actual resume and the target job description.")

if not st.session_state.vectorstore_built:
    st.info("👈 Upload your resume and job description, then build the vector store in the sidebar to begin.")
else:
    tab1, tab2, tab3 = st.tabs(["📝 Interview Q&A", "📄 Extracted JD", "🔍 Retrieved Resume Context"])

    with tab1:
        if st.session_state.answers:
            st.subheader("Your Grounded Prep Guide")
            st.markdown(st.session_state.answers)
            st.download_button(
                "⬇️ Download Prep Guide",
                data=st.session_state.answers,
                file_name="interview_prep_guide.md",
                mime="text/markdown",
            )
        else:
            st.info("Vector store ready! Click 'Generate Interview Q&A' in the sidebar.")

    with tab2:
        if st.session_state.jd_text:
            st.text_area("Job Description Data", st.session_state.jd_text, height=400, disabled=True)

    with tab3:
        if st.session_state.retrieved_chunks:
            for i, doc in enumerate(st.session_state.retrieved_chunks):
                with st.expander(f"Relevant Resume Chunk {i + 1}"):
                    st.write(doc.page_content)
