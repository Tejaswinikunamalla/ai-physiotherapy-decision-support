from ai_predictor import predict_fracture

# Use image inside current project folder
img_path = "temp_xray.jpg"   # or "frac.jpeg"

score, confidence, status = predict_fracture(img_path)

print("AI Risk Score:", score)
print("Confidence:", confidence)
print("Structural Status:", status)