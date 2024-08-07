from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd
import lightgbm as lgb

app = Flask(__name__)

modeleP17 = pickle.load(open('modele_P17.sav', 'rb'))
df_modif_allege = pd.read_csv('df_modif_allege.csv')
sub_X_test = df_modif_allege.drop(columns=['TARGET','index'])

@app.route('/', methods=['GET'])
def home():
    return "Bienvenue à l'API de prédiction de crédit"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        client_id = data['SK_ID_CURR']
        
        client_data = sub_X_test[sub_X_test['SK_ID_CURR'] == client_id]
        
        if client_data.empty:
            return jsonify({'error': 'Client not found'}), 404
        
        prediction = modeleP17.predict(client_data)
        
        result = float(prediction)

        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)