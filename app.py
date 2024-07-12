from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

modeleP17 = pickle.load(open('modele_P17.sav', 'rb'))
df_modif_allege = pd.read_csv('df_modif_allege.csv')
sub_X_test = df_modif_allege.drop(columns=['TARGET'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        client_id = data['SK_ID_CURR']
        
        client_data = sub_X_test[sub_X_test['SK_ID_CURR'] == client_id]
        
        if client_data.empty:
            return jsonify({'error': 'Client not found'}), 404
       
        features = client_data.drop(columns=['SK_ID_CURR']).values.reshape(1, -1)
        
        prediction = modeleP17.predict(features)[0]
        probability = modeleP17.predict_proba(features)[0, 1]
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability)
        }
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)