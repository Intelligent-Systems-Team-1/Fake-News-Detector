import pickle

with open("logistic_model.pkl", "rb") as f:
    vec_log, log_model = pickle.load(f)

with open("xgboost_model.pkl", "rb") as f:
    vec_xgb, xgb_model = pickle.load(f)

# Optional: make sure the vectorizers are the same
assert vec_log.get_feature_names_out().tolist() == vec_xgb.get_feature_names_out().tolist()

combined = {
    "logistic_regression": (vec_log, log_model),
    "xgboost": (vec_xgb, xgb_model)
}

with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(combined, f)

print("✅ Combined models saved to fake_news_model.pkl")
