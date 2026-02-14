import json
import mysql.connector
import numpy as np
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ================== إعدادات عامة ==================
UNIVERSITY_NAME = "جامعة اليرموك"
UNIVERSITY_WEBSITE = "https://www.yu.edu.jo"

TOP_K = 5
SIMILARITY_THRESHOLD = 0.35

st.set_page_config(
    page_title=f"مساعد {UNIVERSITY_NAME} الذكي",
    page_icon="🎓",
    layout="centered"
)

st.title(f"🎓 مساعد {UNIVERSITY_NAME} الذكي")

# ================== مسح المحادثة ==================
if st.button("🧹 مسح المحادثة"):
    st.session_state.messages = []
    st.experimental_rerun()

# ================== OpenRouter ==================
OPENROUTER_API_KEY = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# ================== Embedding Model ==================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embedding_model = load_embedding_model()

# ================== Database ==================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "yuchatbot"
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# ================== Similarity ==================
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ================== Retrieve Context ==================
def retrieve_context(query):
    query_vector = embedding_model.encode(query).tolist()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT title, url, content, embedding
        FROM university_knowledge
    """)
    rows = cursor.fetchall()
    conn.close()

    results = []

    for title, url, content, embedding_json in rows:
        if not embedding_json:
            continue

        try:
            db_vector = json.loads(embedding_json)
            score = cosine_similarity(query_vector, db_vector)

            if score >= SIMILARITY_THRESHOLD:
                results.append({
                    "title": title,
                    "url": url,
                    "content": content,
                    "score": score
                })
        except:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:TOP_K]

# ================== Session ==================
if "messages" not in st.session_state:
    st.session_state.messages = []

# عرض المحادثة السابقة
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ================== User Input ==================
user_input = st.chat_input("اكتب سؤالك عن جامعة اليرموك...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 جاري الاستنتاج..."):

            context_results = retrieve_context(user_input)

            if not context_results:
                answer = (
                    "⚠️ لا أملك معلومات دقيقة للإجابة على هذا السؤال حالياً.\n\n"
                    f"🔗 يمكنك زيارة الموقع الرسمي: {UNIVERSITY_WEBSITE}"
                )

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            else:
                # ===== Context نظيف وموجّه =====
                context_text = "\n\n".join(
                    [
                        f"""
[مصدر رسمي]
العنوان: {r['title']}
المعلومة الأساسية:
{r['content']}
"""
                        for r in context_results
                    ]
                )

                # ===== Prompt استنتاجي =====
                system_prompt = f"""
أنت المساعد الذكي الرسمي لـ {UNIVERSITY_NAME}.

مهمتك:
- فهم سؤال المستخدم حتى لو لم يستخدم نفس الكلمات.
- استنتاج الإجابة من المعنى وليس من التطابق الحرفي.
- دمج المعلومات من أكثر من مصدر إذا لزم.
- صياغة إجابة واضحة ومختصرة.

قواعد:
- اعتمد فقط على المعلومات أدناه.
- لا تنقل النص حرفياً.
- إذا كانت المعلومة غير مكتملة، وضّح ذلك.
- إذا لم تجد المعلومة، قل لا أعلم . إذا لم تكن المعلومة مذكورة حرفياً ولكن يمكن استنتاجها من السياق، فقم بالاستنتاج واذكر ذلك بوضوح.إذا كانت المعلومات ذات صلة قوية بسؤال المستخدم (حتى لو بصياغة مختلفة)، اعتبرها كافية للإجابة.إذا لم تجد إجابة، وجّه المستخدم للموقع الرسمي.

المعلومات المتاحة:
{context_text}

الآن:
أجب عن سؤال المستخدم التالي بإجابة مستنتجة:
"""

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]

                completion = client.chat.completions.create(
                    model="meta-llama/llama-3.1-8b-instruct",
                    messages=messages,
                    temperature=0.2
                )

                answer = completion.choices[0].message.content.strip()
                st.markdown(answer)

                with st.expander("📚 المصادر الرسمية"):
                    for r in context_results:
                        st.markdown(f"- [{r['title']}]({r['url']})")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

