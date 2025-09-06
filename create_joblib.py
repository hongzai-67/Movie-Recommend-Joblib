# create_joblib.py
import argparse
import joblib
import os
from app import IMDBContentBasedRecommendationSystem

def create_joblib(csv_path, out_path="recommender.joblib"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Loading CSV from: {csv_path}")
    rec = IMDBContentBasedRecommendationSystem()
    rec.load_imdb_data(csv_path)         # accepts filepath
    print("Preprocessed. Building content-based system (TF-IDF)...")
    rec.build_content_based_system()
    print("Saving recommender to joblib:", out_path)
    joblib.dump(rec, out_path)
    print("Saved.")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create recommender.joblib from imdb CSV")
    p.add_argument("--csv", "-c", required=True, help="Path to IMDB CSV")
    p.add_argument("--out", "-o", default="recommender.joblib", help="Output joblib path")
    args = p.parse_args()
    create_joblib(args.csv, args.out)
