import os
import time
import json
import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Literal
from google import genai
from google.genai import types

# Must be the very first Streamlit command
st.set_page_config(page_title="Payment Fraud Triaging Agent (Phase 2)", page_icon="🛡️", layout="wide")

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
# 2. TOOLS WITH DOCSTRINGS FOR GEMINI
# ==========================================

def check_ip_risk(ip_address: str) -> dict:
    """Checks IP address risk, country, and VPN/Proxy status.
    
    Args:
        ip_address: The IPv4 address of the user making the transaction.
    """
    if ip_address.startswith("185.") or ip_address.startswith("45."):
        return {"ip": ip_address, "is_vpn": True, "risk_level": "HIGH", "ip_country": "RU"}
    return {"ip": ip_address, "is_vpn": False, "risk_level": "LOW", "ip_country": "US"}

def check_card_bin(card_bin: str) -> dict:
    """Retrieves issuer details and card issuing country based on the 6-digit BIN.
    
    Args:
        card_bin: The first 6 digits of the credit/debit card.
    """
    is_us = card_bin.startswith("4111")
    return {
        "bin": card_bin,
        "brand": "Visa" if card_bin.startswith("4") else "MasterCard",
        "card_type": "Credit",
        "issuing_country": "US" if is_us else "GB"
    }

def check_velocity(user_id: str) -> dict:
    """Checks recent transaction velocity and frequency for the specified user ID in the last hour.
    
    Args:
        user_id: The unique ID of the purchasing account.
    """
    # Simulated velocity lookups
    count = 7 if "suspicious" in user_id else 1
    return {"user_id": user_id, "transactions_last_hour": count}


# Dictionary mapping tool names to python execution references
TOOL_MAP = {
    "check_ip_risk": check_ip_risk,
    "check_card_bin": check_card_bin,
    "check_velocity": check_velocity
}


# ==========================================
# 3. PHASE 2 AGENT ENGINE (ReAct Loop)
# ==========================================

def run_fraud_agent_react(
    tx_id: str,
    user_id: str,
    amount: float,
    currency: str,
    ip_address: str,
    card_bin: str,
    billing_country: str,
    shipping_country: str,
    account_age_days: int,
    api_key: str
) -> tuple[RiskAssessmentOutput, list]:
    
    client = genai.Client(api_key=api_key)
    trace_logs = []

    system_prompt = """
    You are an expert FinTech Payment Fraud Triaging Agent running in an autonomous ReAct loop.
    You have access to tools: check_ip_risk, check_card_bin, and check_velocity.

    WORKFLOW:
    1. Inspect the incoming transaction request.
    2. Dynamically execute tools to collect necessary intelligence.
    3. Evaluate risk rules:
       - IP VPN/Proxy (+0.35)
       - Velocity > 3 tx/hr (+0.30)
       - Country mismatches between Billing, Shipping, and IP Country (+0.25 per mismatch)
       - Account age < 7 days with amount > $500 (+0.20)
    4. Provide the final decision as structured output.
       - risk_score < 0.35 => APPROVE
       - 0.35 <= risk_score < 0.70 => CHALLENGE_3DS
       - risk_score >= 0.70 => BLOCK
    """

    user_context = f"""
    INCOMING TRANSACTION:
    - ID: {tx_id}
    - User ID: {user_id}
    - Amount: ${amount:.2f} {currency}
    - Account Age: {account_age_days} days
    - Billing Country: {billing_country}
    - Shipping Country: {shipping_country}
    - IP Address: {ip_address}
    - Card BIN: {card_bin}
    """

    models_to_try = ['gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-1.5-flash']
    
    for model_name in models_to_try:
        try:
            # Step 1: Initial chat turn with tools enabled
            chat = client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[check_ip_risk, check_card_bin, check_velocity],
                    temperature=0.1
                )
            )

            trace_logs.append(f"🤖 Initializing ReAct Agent with model `{model_name}`...")
            response = chat.send_message(user_context)

            # ReAct Execution Loop (Handles dynamic function calls)
            max_turns = 5
            for turn in range(max_turns):
                # Check if Gemini requested function calls
                if response.function_calls:
                    for call in response.function_calls:
                        func_name = call.name
                        func_args = call.args
                        trace_logs.append(f"⚙️ **Tool Call Requested**: `{func_name}({json.dumps(func_args)})`")

                        # Execute local python function
                        if func_name in TOOL_MAP:
                            result = TOOL_MAP[func_name](**func_args)
                            trace_logs.append(f"📥 **Tool Result Received**: `{json.dumps(result)}`")

                            # Feed tool result back into the chat session
                            response = chat.send_message(
                                types.Content(
                                    parts=[
                                        types.Part.from_function_response(
                                            name=func_name,
                                            response={"result": result}
                                        )
                                    ]
                                )
                            )
                        else:
                            trace_logs.append(f"❌ Unknown tool requested: {func_name}")
                else:
                    # Model done gathering tool information; request final structured output
                    trace_logs.append("🧠 Tool collection finished. Requesting final structured decision...")
                    final_response = chat.send_message(
                        "Formulate final risk assessment strictly matching JSON structure.",
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=RiskAssessmentOutput
                        )
                    )
                    parsed_json = json.loads(final_response.text)
                    return RiskAssessmentOutput(**parsed_json), trace_logs

        except Exception as e:
            trace_logs.append(f"⚠️ Model {model_name} failed: {str(e)}")
            continue

    raise Exception("All models failed to complete ReAct loop.")


# ==========================================
# 4. STREAMLIT UI LAYOUT
# ==========================================

st.title("🛡️ Autonomous Payment Fraud Triaging Agent")
st.caption("Phase 2: Dynamic Function Calling & ReAct Orchestration Loop")

with st.sidebar:
    st.header("🔑 API Configuration")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get your key at aistudio.google.com"
    )

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
    elif preset == "🔴 High-Risk Scenario":
        tx_id_val, user_id_val = "TX_9902", "usr_suspicious_99"
        amount_val, account_age_val = 1250.00, 2
        ip_val, bin_val = "185.220.101.5", "411111"
        billing_val, shipping_val = "US", "GB"
    else:
        tx_id_val, user_id_val = "TX_0001", "usr_custom"
        amount_val, account_age_val = 150.00, 30
        ip_val, bin_val = "192.168.1.1", "411111"
        billing_val, shipping_val = "US", "US"

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

    run_btn = st.button("🚀 Run Autonomous ReAct Agent", type="primary", use_container_width=True)

with col2:
    st.subheader("📊 Agent Decision Output")
    
    if run_btn:
        if not api_key_input:
            st.error("Please provide your Gemini API key.")
        else:
            with st.spinner("Executing ReAct Reasoning & Tool Calling Loop..."):
                try:
                    result, trace_logs = run_fraud_agent_react(
                        tx_id=tx_id,
                        user_id=user_id,
                        amount=amount,
                        currency="USD",
                        ip_address=ip_address,
                        card_bin=card_bin,
                        billing_country=billing_country,
                        shipping_country=shipping_country,
                        account_age_days=account_age,
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

                    # Trace Log UI Component
                    with st.expander("🧩 View Agent Execution Trace (ReAct Log)", expanded=True):
                        for log in trace_logs:
                            st.markdown(log)

                except Exception as e:
                    st.error(f"Evaluation Error: {e}")
    else:
        st.info("Fill out inputs on the left and click 'Run Autonomous ReAct Agent'.")