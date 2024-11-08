import csv
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

def load_activities(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        return [(row[1], row[2]) for row in reader] #This extracts the activity_id and the activity name

def cluster_activities(activities, model, eps=0.3, min_samples=2, min_cluster_size=2):
    # Extract activity names 
    activity_names = [name for _, name in activities]
    
    # Generate embeddings for semantic understanding
    embeddings = model.encode(activity_names)
    
    # Perform clustering using DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(embeddings)
    
    # Group activities by cluster
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(activities[i])

    final_clusters = []
    singleton_cluster = [] #Some activies may appear single and very unique
    for cluster in clusters.values():
        if len(cluster) >= min_cluster_size:
            final_clusters.append(cluster)
        else:
            singleton_cluster.extend(cluster)

    # Add each singleton as its own cluster
    for activity in singleton_cluster:
        final_clusters.append([activity])

    return final_clusters

def save_clusters(clusters, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Cluster ID', 'Activity ID', 'Activity Name'])
        for cluster_id, cluster in enumerate(clusters, 1):
            for activity_id, name in cluster:
                writer.writerow([cluster_id, activity_id, name])

def main(input_file, output_file):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    activities = load_activities(input_file)
    clusters = cluster_activities(activities, model)
    save_clusters(clusters, output_file)

    # Check the data and verify
    print(f"Input activities: {len(activities)}")
    print(f"Output clusters: {len(clusters)}")
    print(f"Activities in clusters: {sum(len(cluster) for cluster in clusters)}")

if __name__ == '__main__':
    input_file = 'input.csv'
    output_file = 'output.csv'
    main(input_file, output_file)