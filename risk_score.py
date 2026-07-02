def calculate_risk_score(
    age_group,
    pain_level,
    mobility,
    post_surgery,
    swelling
):
    score = 0
    max_score = 15

    # Age
    if age_group == "> 60":
        score += 2
    elif age_group == "41 – 60":
        score += 1

    # Pain
    if pain_level >= 7:
        score += 3
    elif pain_level >= 4:
        score += 2

    # Mobility
    if mobility:
        score += 4

    # Surgery
    if post_surgery:
        score += 3

    # Swelling
    if swelling:
        score += 2

    risk_percentage = int((score / max_score) * 100)

    if risk_percentage < 30:
        category = "Low"
        recommendation = "Routine monitoring recommended"
    elif risk_percentage < 60:
        category = "Moderate"
        recommendation = "Supervised Physiotherapy Suggested"
    else:
        category = "High"
        recommendation = "Physiotherapy Evaluation Strongly Recommended"

    confidence = min(95, 80 + score)

    return risk_percentage, category, recommendation, confidence