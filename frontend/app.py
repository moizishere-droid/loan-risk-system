import os
import streamlit as st
import requests

# CONFIG
# Reads from environment — localhost for dev, http://api:8000 for Docker
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title = "Loan Risk Assessment System",
    page_icon  = "🏦",
    layout     = "wide"
)

st.title("🏦 Loan Risk Assessment System")
st.markdown("---")


# HELPERS
LOAN_GRADES    = ["A", "B", "C", "D", "E", "F", "G"]
LOAN_INTENTS   = ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT","MEDICAL", "PERSONAL", "VENTURE"]
HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE", "OTHER"]


def show_prediction_result(data):
    """Display prediction result — decision, rate, cluster, reasons, SHAP."""
    cls     = data["classification"]
    reg     = data.get("regression")
    clu     = data["clustering"]
    plain   = data.get("plain_reasons", {})
    wf      = data["workflow"]
    pred_id = data["prediction_id"]

    st.markdown(f"**Prediction ID:** #{pred_id} | **Workflow:** {wf.replace('_', ' ').title()}")
    st.markdown("---")

    # Decision
    decision = cls["decision"]
    prob     = round(cls["default_probability"] * 100, 2)
    if decision == "APPROVED":
        st.success(f" APPROVED — Default Risk: {prob}%")
    else:
        st.error(f" REJECTED — Default Risk: {prob}%")

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", f"{prob}%")
    col2.metric("Threshold",           f"{round(cls['threshold'] * 100)}%")
    if reg:
        col3.metric("Predicted Interest Rate", f"{round(reg['predicted_interest_rate'], 2)}%")

    st.markdown("---")

    # Cluster
    st.markdown(f"**Borrower Segment:** {clu['cluster_label']}")
    st.markdown(
        f"High Value Borrower: {round(clu['probabilities']['High Value Borrower'] * 100, 1)}% "
        f"| Standard Borrower: {round(clu['probabilities']['Standard Borrower'] * 100, 1)}%"
    )

    st.markdown("---")

    # Plain reasons
    if plain.get("decision_reasons"):
        st.markdown("**Decision Reasons:**")
        for i, r in enumerate(plain["decision_reasons"], 1):
            st.markdown(f"{i}. {r}")

    if reg and plain.get("rate_reasons"):
        st.markdown("**Interest Rate Reasons:**")
        for i, r in enumerate(plain["rate_reasons"], 1):
            st.markdown(f"{i}. {r}")

    st.markdown("---")

    # SHAP — Classification
    st.markdown("**SHAP — Classification:**")
    shap_cls = (
        [{"Feature": r["readable_name"], "SHAP Impact": r["shap_impact"], "Direction": r["direction"]}
         for r in cls["reasons_for_default"]] +
        [{"Feature": r["readable_name"], "SHAP Impact": r["shap_impact"], "Direction": r["direction"]}
         for r in cls["reasons_against_default"]]
    )
    st.table(shap_cls)

    # SHAP — Regression
    if reg:
        st.markdown("**SHAP — Regression:**")
        shap_reg = (
            [{"Feature": r["readable_name"], "SHAP Impact": r["shap_impact"], "Direction": r["direction"]}
             for r in reg["reasons_high_rate"]] +
            [{"Feature": r["readable_name"], "SHAP Impact": r["shap_impact"], "Direction": r["direction"]}
             for r in reg["reasons_low_rate"]]
        )
        st.table(shap_reg)


def input_form(with_rate=False, key_prefix=""):
    """Shared input form for both workflows."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Info**")
        cnic          = st.text_input("CNIC", placeholder="42101-1234567-1", key=f"{key_prefix}_cnic")
        person_age    = st.number_input("Age", min_value=18, max_value=99,   value=30,      key=f"{key_prefix}_age")
        person_income = st.number_input("Annual Income ($)", min_value=1.0,  value=50000.0, key=f"{key_prefix}_income", step=1000.0)
        person_emp    = st.number_input("Employment Length (years)", min_value=0.0, value=5.0, key=f"{key_prefix}_emp", step=0.5)
        home_own      = st.selectbox("Home Ownership", HOME_OWNERSHIP, key=f"{key_prefix}_home")

    with col2:
        st.markdown("**Loan Details**")
        loan_amnt    = st.number_input("Loan Amount ($)", min_value=1.0, value=10000.0, key=f"{key_prefix}_amnt", step=500.0)
        loan_grade   = st.selectbox("Loan Grade", LOAN_GRADES,   key=f"{key_prefix}_grade")
        loan_intent  = st.selectbox("Loan Purpose", LOAN_INTENTS, key=f"{key_prefix}_intent")
        loan_pct_inc = st.number_input("Loan % of Income (0-1)", min_value=0.01, max_value=0.99, value=0.20, key=f"{key_prefix}_pct", step=0.01)
        loan_rate    = None
        if with_rate:
            loan_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=50.0, value=12.0, key=f"{key_prefix}_rate", step=0.1)

    with col3:
        st.markdown("**Credit Info**")
        default_on_file = st.selectbox("Previous Default?", ["N", "Y"],
                                       format_func=lambda x: "No" if x == "N" else "Yes",
                                       key=f"{key_prefix}_default")
        cred_hist       = st.number_input("Credit History (years)", min_value=0.0, value=5.0, key=f"{key_prefix}_cred", step=0.5)

    payload = {
        "cnic"                      : cnic,
        "loan_amnt"                 : float(loan_amnt),
        "loan_grade"                : loan_grade,
        "loan_intent"               : loan_intent,
        "loan_percent_income"       : float(loan_pct_inc),
        "person_income"             : float(person_income),
        "person_age"                : int(person_age),
        "person_emp_length"         : float(person_emp),
        "person_home_ownership"     : home_own,
        "cb_person_default_on_file" : default_on_file,
        "cb_person_cred_hist_length": float(cred_hist),
    }
    if with_rate and loan_rate:
        payload["loan_int_rate"] = float(loan_rate)

    return payload


# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "🆕 New Applicant",
    "🔄 Existing Loan",
    "👤 Applicant Profile",
    "🗑️ Delete Applicant"
])


# TAB 1 — NEW APPLICANT
with tab1:
    st.subheader("New Applicant")
    st.markdown("No interest rate yet — system will predict it.")
    st.markdown("---")

    payload = input_form(with_rate=False, key_prefix="new")

    if st.button("Run Prediction", key="btn_new"):
        if not payload["cnic"]:
            st.warning("Please enter CNIC.")
        else:
            with st.spinner("Running prediction..."):
                try:
                    r = requests.post(f"{API_URL}/predict/new", json=payload, timeout=60)
                    if r.status_code == 200:
                        show_prediction_result(r.json())
                    else:
                        st.error(f"Error: {r.json().get('detail', r.text)}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")


# TAB 2 — EXISTING LOAN
with tab2:
    st.subheader("Existing Loan")
    st.markdown("Interest rate already assigned — regression skipped.")
    st.markdown("---")

    payload = input_form(with_rate=True, key_prefix="existing")

    if st.button("Run Prediction", key="btn_existing"):
        if not payload["cnic"]:
            st.warning("Please enter CNIC.")
        else:
            with st.spinner("Running prediction..."):
                try:
                    r = requests.post(f"{API_URL}/predict/existing", json=payload, timeout=60)
                    if r.status_code == 200:
                        show_prediction_result(r.json())
                    else:
                        st.error(f"Error: {r.json().get('detail', r.text)}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")


# TAB 3 — APPLICANT PROFILE
with tab3:
    st.subheader("Applicant Profile")
    st.markdown("Enter CNIC to view full profile and last visit details.")
    st.markdown("---")

    cnic_search = st.text_input("CNIC", placeholder="42101-1234567-1", key="cnic_search")

    if st.button("Search", key="btn_search"):
        if not cnic_search:
            st.warning("Please enter CNIC.")
        else:
            with st.spinner("Fetching..."):
                try:
                    r = requests.get(f"{API_URL}/applicant/{cnic_search}", timeout=30)
                    if r.status_code == 200:
                        d = r.json()

                        # Summary
                        st.markdown("### Applicant Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Visits",  d["total_visits"])
                        col2.metric("Approved",       d["total_approved"])
                        col3.metric("Rejected",       d["total_rejected"])
                        col4.metric("Last Decision",  d["last_decision"])

                        st.markdown(f"**First Seen:** {d['first_seen']}  |  **Last Seen:** {d['last_seen']}")
                        st.markdown(f"**Last Loan Amount:** ${d['last_loan_amnt']:,.0f}  |  **Last Rate:** {d['last_interest_rate']}%")

                        st.markdown("---")

                        # Last visit
                        lv = d.get("last_visit")
                        if lv:
                            st.markdown("### Last Visit")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Prediction ID",       f"#{lv['prediction_id']}")
                            col2.metric("Decision",             lv["decision"])
                            col3.metric("Default Probability",  f"{round(lv['default_probability'] * 100, 2)}%")

                            # Inputs
                            if lv.get("inputs"):
                                st.markdown("**Inputs Submitted:**")
                                st.table([lv["inputs"]])

                            # Explanations
                            if lv.get("explanations"):
                                st.markdown("**SHAP Explanations:**")
                                exp_rows = [
                                    {
                                        "Task"       : e["task"],
                                        "Feature"    : e["readable_name"],
                                        "SHAP Impact": e["shap_impact"],
                                        "Direction"  : e["direction"]
                                    }
                                    for e in lv["explanations"]
                                ]
                                st.table(exp_rows)

                    elif r.status_code == 404:
                        st.warning(f"No applicant found with CNIC: {cnic_search}")
                    else:
                        st.error(f"Error: {r.json().get('detail', r.text)}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")


# TAB 4 — DELETE APPLICANT
with tab4:
    st.subheader("Delete Applicant")
    st.markdown("Permanently delete an applicant and all their predictions, inputs, and explanations.")
    st.markdown("---")

    cnic_delete = st.text_input("CNIC", placeholder="42101-1234567-1", key="cnic_delete")
    st.warning(" This action cannot be undone. All records will be permanently deleted.")

    if st.button("Delete", key="btn_delete", type="primary"):
        if not cnic_delete:
            st.warning("Please enter CNIC.")
        else:
            with st.spinner("Deleting..."):
                try:
                    r = requests.delete(f"{API_URL}/applicant/{cnic_delete}", timeout=30)
                    if r.status_code == 200:
                        st.success(r.json().get("message", "Deleted successfully."))
                    elif r.status_code == 404:
                        st.warning(f"No applicant found with CNIC: {cnic_delete}")
                    else:
                        st.error(f"Error: {r.json().get('detail', r.text)}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")