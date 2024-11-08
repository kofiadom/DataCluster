# Home Assignment
### Submitted by Kofi Adom.

This assignment is a solution for clustering similar activities based on their names using Natural Language Processing (NLP) techniques. It uses the Sentence Transformers library to generate semantic embeddings of activity names and the DBSCAN algorithm for clustering.

## How The Solution Works

1. The script reads the input CSV file and extracts the activity IDs and names.
2. It uses a Sentence Transformer model (all-MiniLM-L6-v2) to convert each activity name into a vector representation.
3. The DBSCAN algorithm is applied to these vectors to identify clusters of similar activities.
4. Activities that don't fit into any cluster are assigned their own single-activity clusters.
5. The results are written to the output CSV file, with each activity assigned a cluster ID.

## Requirements

- Python 3.7+
- sentence-transformers
- scikit-learn

## Set-up

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the script:
   ```
   python main.py
   ```

2. The script will generate an output file named clustered_activities.csv with the following columns:
   - Cluster ID
   - Activity ID
   - Activity Name



