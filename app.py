from flask import Flask, request, jsonify
import pickle
import pandas as pd
import lightgbm as lgb

app = Flask(__name__)

with open('modele_P17.sav', 'rb') as model_file:
    modeleP17 = pickle.load(model_file)

df_modif_allege = pd.read_csv('df_modif_allege.csv')
sub_X_test = df_modif_allege.drop(columns=['TARGET'])

@app.route('/')
def home():
    return "Bienvenue à l'API de prédiction de crédit"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({'error': 'Requête non JSON'}), 400
        
        data = request.get_json()
        
        if 'SK_ID_CURR' not in data:
            return jsonify({'error': 'SK_ID_CURR non fourni'}), 400
        
        client_id = data['SK_ID_CURR']
        
        client_data = sub_X_test[sub_X_test['SK_ID_CURR'] == client_id]
        
        if client_data.empty:
            return jsonify({'error': 'Client non trouvé'}), 404
        
        features = client_data.drop(columns=['SK_ID_CURR']).values.reshape(1, -1)
        
        prediction = modeleP17.predict(features)[0]
        probability = modeleP17.predict_proba(features)[0, 1]
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability)
        }
        return jsonify(result)
    
    except KeyError as e:
        return jsonify({'error': f'Clé manquante dans les données: {str(e)}'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'API fonctionne correctement'}), 200

if __name__ == '__main__':
    app.run(debug=True)