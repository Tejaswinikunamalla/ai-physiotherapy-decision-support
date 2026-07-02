# def fusion_engine(xray_result, clinical_score):
#     """
#     Combines X-ray result and clinical risk score
#     to produce overall rehabilitation likelihood.
#     """

#     # ---- Map X-ray result to AI score ----
#     if xray_result == "Abnormal":
#         ai_score = 80
#     else:  # Normal
#         ai_score = 10

#     # ---- Weighted fusion ----
#     final_score = (0.6 * ai_score) + (0.4 * clinical_score)
#     final_score = int(final_score)

#     # ---- Safety gating ----
#     if xray_result == "Normal" and clinical_score < 30:
#         final_score = min(final_score, 29)

#     # ---- Category & Recommendation ----
#     if final_score < 30:
#         category = "Low"
#         recommendation = "Monitoring recommended"
#     elif final_score < 60:
#         category = "Moderate"
#         recommendation = "Clinical evaluation recommended"
#     else:
#         category = "High"
#         recommendation = "Physiotherapy consultation advised"

#     return final_score, category, recommendation
def fusion_engine(ai_score, clinical_score):
    """
    Combines AI structural risk score and clinical risk score
    to produce overall rehabilitation likelihood.
    """

    # ---- Weighted fusion ----
    # final_score = (0.55 * ai_score) + (0.45 * clinical_score)
    # final_score = int(final_score)

    # ---- Safety gating ----
    final_score=0
    if ai_score < 30 and clinical_score < 30:
        final_score = min(final_score, 29)
    elif ai_score<30 and clinical_score>=75:
        final_score=clinical_score*0.85
    elif ai_score<30:
        final_score=(0.3 * ai_score) + (0.7 * clinical_score)
    else:
        if clinical_score<30:
            final_score=(0.5 * ai_score) + (0.5 * clinical_score)
        else:
            final_score=(0.6 * ai_score) + (0.4 * clinical_score)
    final_score = int(final_score)

    # ---- Category & Recommendation ----
    if final_score < 30:
        category = "Low"
        recommendation = "Low structural and clinical risk. Home care and symptom observation advised."
    elif final_score < 63:
        category = "Moderate"
        recommendation = "A structured physiotherapy assessment is recommended to prevent functional decline."
    else:
        category = "High"
        recommendation = "Early physiotherapy consultation and structured rehabilitation planning are strongly recommended."

    return final_score, category, recommendation