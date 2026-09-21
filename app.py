import os
import time
import json
import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Literal
from google import genai
from google.genai import types

# Must be the very first Streamlit command
st.set_page_config(page_title="Payment Fraud Triaging Agent", page_icon="🛡️", layout="wide")

# ==========================================
# 1. SCHEMAS (Structured Output Models)
# ==========================================

class RiskAssessmentOutput(BaseModel):
    transaction_id: str
    risk_score: float = Field(..., description="Calculated risk score from 0.0 to 1.0")
    decision: Literal["APPROVE", "CHALLENGE_3DS", "BLOCK"]
    detected_anomalies: List[str]
    audit_explanation: str


# ==========================================
# 2. DETERMINISTIC MOCK TOOLS
# ==========================================

def check_ip_risk(ip_address: str) -> dict:
    if ip_address.startswith("185.") or ip_address.startswith("45."):
        return {"ip": ip_address, "is_vpn": True, "risk_level": "HIGH", "ip_country": "RU"}
    return {"ip": ip_address, "is_vpn": False, "risk_level": "LOW", "ip_country": "US"}

def check_card_bin(card_bin: str) -> dict:
    is_us = card_bin.startswith("4111")
    return {
        "bin": card_bin,
        "brand": "Visa" if card_bin.startswith("4") else "MasterCard",
        "card_type": "Credit",
        "issuing_country": "US" if is_us else "GB"
    }

def check_velocity(user_id: str, is_suspicious_preset: bool = False) -> dict:
    count = 7 if is_suspicious_preset else 1
    return {"user_id": user_id, "transactions_last_hour": count}


# ==========================================
# 3. AGENT CORE ENGINE (with 503 Retry & Fallbacks)
# ==========================================

def run_fraud_agent(
    tx_id: str,
    user_id: str,
    amount: float,
    currency: str,
    ip_address: str,
    card_bin: str,
    billing_country: str,
    shipping_country: str,
    account_age_days: int,
    is_suspicious_preset: bool,
    api_key: str
) -> RiskAssessmentOutput:
    
    client = genai.Client(api_key=api_key)

    # 1. Execute deterministic tool checks
    ip_info = check_ip_risk(ip_address)
    bin_info = check_card_bin(card_bin)
    velocity_info = check_velocity(user_id, is_suspicious_preset)

    system_prompt = """
    You are an expert FinTech Payment Fraud Triaging Agent.
    Evaluate the incoming transaction using transaction details and tool signals.

    STRICT RISK EVALUATION RULES:
    1. Score Range:
       - risk_score < 0.35 => APPROVE
       - 0.35 <= risk_score < 0.70 => CHALLENGE_3DS
       - risk_score >= 0.70 => BLOCK
    
    2. Escalation Factors:
       - IP flagged as VPN/Proxy (+0.35)
       - High velocity (>3 tx/hr) (+0.30)
       - Country Mismatches (+0.25 per mismatch)
       - Account age < 7 days on high amount (> $500) (+0.20)

    Return valid JSON strictly matching the requested schema.
    """

    user_context = f"""
    TRANSACTION DATA:
    - ID: {tx_id}
    - User ID: {user_id}
    - Amount: ${amount:.2f} {currency}
    - Account Age: {account_age_days} days
    - Billing Country: {billing_country}
    - Shipping Country: {shipping_country}

    TOOL RESULTS:
    - IP Tool: {json.dumps(ip_info)}
    - Card BIN Tool: {json.dumps(bin_info)}
    - Velocity Tool: {json.dumps(velocity_info)}
    """

    # Model priority list
    models_to_try = [
        'gemini-3.6-flash',
        'gemini-3.5-flash'    ]

    last_exception = None

    for model_name in models_to_try:
        # Retry up to 3 times per model if hit with temporary 503 capacity errors
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_context,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=RiskAssessmentOutput,
                        temperature=0.1,
                    ),
                )
                parsed_json = json.loads(response.text)
                return RiskAssessmentOutput(**parsed_json)

            except Exception as e:
                last_exception = e
                err_msg = str(e)
                
                # Handle 503 or UNAVAILABLE errors with exponential backoff delay
                if "503" in err_msg or "UNAVAILABLE" in err_msg:
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    # Non-503 error, proceed directly to the next model in list
                    break

    raise last_exception


# ==========================================
# 4. STREAMLIT UI LAYOUT
# ==========================================

st.title("🛡️ Autonomous Payment Fraud Triaging Agent")
st.caption("Phase 1: Deterministic Tool Calling + Structured Reasoning Engine")

# Sidebar Configuration
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get your key at aistudio.google.com"
    )

# Main Form Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Transaction Inputs")
    
    preset = st.radio(
        "Load Quick Scenario Preset:",
        ["Custom Input", "🟢 Safe Scenario", "🔴 High-Risk Scenario"],
        horizontal=True
    )

    if preset == "🟢 Safe Scenario":
        tx_id_val, user_id_val = "TX_1001", "usr_legit_42"
        amount_val, account_age_val = 49.99, 180
        ip_val, bin_val = "12.180.12.1", "411111"
        billing_val, shipping_val = "US", "US"
        is_suspicious_preset = False
    elif preset == "🔴 High-Risk Scenario":
        tx_id_val, user_id_val = "TX_9902", "usr_suspicious_99"
        amount_val, account_age_val = 1250.00, 2
        ip_val, bin_val = "185.220.101.5", "411111"
        billing_val, shipping_val = "US", "GB"
        is_suspicious_preset = True
    else:
        tx_id_val, user_id_val = "TX_0001", "usr_custom"
        amount_val, account_age_val = 150.00, 30
        ip_val, bin_val = "192.168.1.1", "411111"
        billing_val, shipping_val = "US", "US"
        is_suspicious_preset = False

    tx_id = st.text_input("Transaction ID", tx_id_val)
    user_id = st.text_input("User ID", user_id_val)
    amount = st.number_input("Amount ($)", value=amount_val, step=10.0)
    account_age = st.number_input("Account Age (Days)", value=account_age_val)
    ip_address = st.text_input("IP Address", ip_val)
    card_bin = st.text_input("Card BIN (First 6 digits)", bin_val)
    
    c_a, c_b = st.columns(2)
    with c_a:
        billing_country = st.text_input("Billing Country", billing_val)
    with c_b:
        shipping_country = st.text_input("Shipping Country", shipping_val)

    run_btn = st.button("🚀 Analyze Transaction Risk", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Agent Decision Output")
    
    if run_btn:
        if not api_key_input:
            st.error("Please enter your Google Gemini API Key in the sidebar.")
        else:
            with st.spinner("Analyzing risk parameters..."):
                try:
                    result = run_fraud_agent(
                        tx_id=tx_id,
                        user_id=user_id,
                        amount=amount,
                        currency="USD",
                        ip_address=ip_address,
                        card_bin=card_bin,
                        billing_country=billing_country,
                        shipping_country=shipping_country,
                        account_age_days=account_age,
                        is_suspicious_preset=is_suspicious_preset,
                        api_key=api_key_input
                    )

                    if result.decision == "APPROVE":
                        st.success(f"### Decision: {result.decision}")
                    elif result.decision == "CHALLENGE_3DS":
                        st.warning(f"### Decision: {result.decision}")
                    else:
                        st.error(f"### Decision: {result.decision}")

                    st.metric(label="Risk Score", value=f"{result.risk_score:.2f} / 1.00")

                    st.markdown("**Anomalies:**")
                    if result.detected_anomalies:
                        for anomaly in result.detected_anomalies:
                            st.write(f"- ⚠️ {anomaly}")
                    else:
                        st.write("None")

                    st.markdown("**Explanation:**")
                    st.info(result.audit_explanation)

                    with st.expander("🔍 View Raw Structured Output"):
                        st.json(result.model_dump())

                except Exception as e:
                    st.error(f"Evaluation Error: {e}")
    else:
        st.info("Fill out inputs on the left and click 'Analyze Transaction Risk'.")