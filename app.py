from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from ahp import AHP
import graphviz
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def get_filtered_recommendations(score, criteria, kelurahan):
    # Baca file CSV
    df = pd.read_csv('smpn_bogor.csv', delimiter=';')

    df = df.dropna()  # Menghapus baris dengan nilai yang hilang
    df = df[df['SKOR RAPOT'] > 0]

    if criteria == 'fasilitas':
        # Filter data berdasarkan skor rapot dan hanya yang memiliki fasilitas
        df_filtered = df[df['SKOR RAPOT'] <= score]
        df_filtered = df_filtered[(df_filtered['PERPUSTAKAAN'] > 0) & 
                                  (df_filtered['LAPANGAN'] > 0) & 
                                  (df_filtered['LABORATORIUM '] > 0) & 
                                  (df_filtered['UKS'] > 0)]
    else:
        # Filter data berdasarkan kelurahan dan skor rapot
        df_filtered = df[df['KELURAHAN'].str.contains(kelurahan, case=False, na=False)]
        df_filtered = df_filtered[df_filtered['SKOR RAPOT'] <= score]

    if df_filtered.empty:
        return "No schools found for the given criteria."

    features = df_filtered[['ZONA', 'PERPUSTAKAAN', 'LAPANGAN', 'LABORATORIUM ', 'UKS']]

    if len(features) < 2:
        return "No schools found for the given criteria."

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=2, random_state=42)
    df_filtered['Cluster'] = kmeans.fit_predict(features_scaled)

    if criteria == 'Zona':
        pairwise_matrix = np.array([
            [1, 2, 3, 4, 5],
            [0.5, 1, 2, 3, 4],
            [0.333, 0.5, 1, 2, 3],
            [0.25, 0.333, 0.5, 1, 2],
            [0.2, 0.25, 0.333, 0.5, 1]
        ])
    elif criteria == 'fasilitas':
        pairwise_matrix = np.array([
            [1, 0.5, 0.333, 0.25, 0.2],
            [2, 1, 0.5, 0.333, 0.25],
            [3, 2, 1, 0.5, 0.333],
            [4, 3, 2, 1, 0.5],
            [5, 4, 3, 2, 1]
        ])
    
    ahp = AHP(pairwise_matrix)
    
    if ahp.rasio_konsistensi >= 0.1:
        return "Consistency ratio is too high. Please revise the pairwise comparison matrix."
    
    ahp_scores = ahp.nilai_alternatif(features.values)
    df_filtered['AHP_Score'] = ahp_scores

    df_filtered = df_filtered.drop_duplicates(subset='SMP NEGERI')
    sorted_df = df_filtered.sort_values(by='AHP_Score', ascending=False).head(10)

    return sorted_df

def create_ahp_hierarchy(recommendations):
    dot = graphviz.Digraph(comment='AHP Hierarchy')
    
    # Add nodes for the hierarchy
    dot.node('A', 'Tujuan: Memilih Sekolah Terbaik')
    dot.node('B', 'Kriteria: Zona')
    dot.node('C', 'Kriteria: Fasilitas')
    
    # Connect nodes
    dot.edges(['AB', 'AC'])
    
    # Add nodes for each recommended school
    for index, row in recommendations.iterrows():
        school_node = f"D{index}"
        school_label = f"{row['SMP NEGERI']}\nSkor AHP: {row['AHP_Score']:.4f}"
        dot.node(school_node, school_label)
        dot.edge('B', school_node)
        dot.edge('C', school_node)

    # Save the diagram
    filepath = 'static/ahp_hierarchy'
    dot.render(filepath, format='png', cleanup=True)
    return filepath + '.png'

@app.route('/recommendation', methods=['POST'])
def recommendation():
    try:
        score = float(request.form['score'].replace(',', '.'))
        criteria = request.form['criteria']
        kelurahan = request.form['kelurahan']
    except ValueError:
        return "Invalid input. Please ensure all fields are filled correctly."
    
    recommendations = get_filtered_recommendations(score, criteria, kelurahan)
    
    if isinstance(recommendations, str):
        return recommendations
    
    recommendations = recommendations[['SMP NEGERI', 'KELURAHAN', 'SKOR RAPOT', 'AHP_Score']]
    table_html = recommendations.to_html(classes='table recommendation-table', index=False)
    
    hierarchy_path = create_ahp_hierarchy(recommendations)
    
    return render_template('recommendation.html', table_html=table_html, criteria=criteria, kelurahan=kelurahan, hierarchy_path=hierarchy_path)

@app.route('/static/<path:filename>')
def send_image(filename):
    return send_file(os.path.join('static', filename))

if __name__ == '__main__':
    app.run(debug=True)
