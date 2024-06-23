import streamlit as st
import pandas as pd
import shap
import lightgbm as lgb
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

modeleP17 = pickle.load(open('modele_P17.sav', 'rb'))
df_modif_allege = pd.read_csv('df_modif_allege.csv')

train, test = train_test_split(df_modif_allege, test_size=0.2, random_state=0)
train.drop(columns=['index'], inplace=True)
test.drop(columns=['index'], inplace=True)
X_train = train.drop(columns=['TARGET'])
y_train = train['TARGET']
X_test = test.drop(columns=['TARGET'])
y_true = test['TARGET']

sub_X_test = X_test.copy()

explainer_glob = shap.TreeExplainer(modeleP17)
shap_values_glob = explainer_glob.shap_values(sub_X_test)

global_importance_scores = abs(shap_values_glob).mean(axis=0)
importance_df = pd.DataFrame({'Feature': sub_X_test.columns, 'Importance': global_importance_scores})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
top_20_features = importance_df.head(20)['Feature'].tolist()

selected_client_id = st.selectbox('Selectionnez un client :', sub_X_test['SK_ID_CURR'])
random_observation_df = sub_X_test[sub_X_test['SK_ID_CURR'] == selected_client_id]

explainer = shap.TreeExplainer(modeleP17)
shap_values = explainer.shap_values(random_observation_df)
y_pred_lim = (modeleP17.predict(random_observation_df) >= 0.4).astype(int)
y_pred_proba = modeleP17.predict(random_observation_df)

client_id = random_observation_df['SK_ID_CURR'].values[0]

st.title(f"Dashboard de Credit")
st.markdown(f"### Num client: {client_id}")

credit_accepted = 'Non' if y_pred_lim[0] == 1 else 'Oui'
st.markdown(f"### Credit accepte: {credit_accepted}")

st.markdown(f"### Score du client")
st.markdown(f"Score: {y_pred_proba[0]:.2f}")
threshold = 0.4
fig, ax = plt.subplots(figsize=(6, 1))
ax.barh([0], [y_pred_proba[0]], color=['green' if y_pred_proba[0] < threshold else 'red'])
ax.axvline(x=threshold, color='blue', linestyle='--')
ax.set_xlim(0, 1)
ax.set_yticks([])
ax.set_xticks([0, threshold, 1])
ax.set_xticklabels(['0', '0.4', '1'])
st.pyplot(fig)

st.markdown("### Feature Importance Globale")
fig_summary, ax_summary = plt.subplots()
shap.summary_plot(shap_values_glob, sub_X_test, plot_type="dot", show=False)
st.pyplot(fig_summary)

def st_shap(plot, height=None):
    """Display a SHAP plot in Streamlit."""
    import streamlit.components.v1 as components
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.markdown("### Feature Importance Locale")
fig_force = shap.force_plot(explainer.expected_value, shap_values[0, :], random_observation_df.iloc[0, :])
st_shap(fig_force, height=150)

local_importance = abs(shap_values).mean(axis=0)
local_importance_df = pd.DataFrame({'Feature': random_observation_df.columns, 'Importance Relative': local_importance})
local_importance_df = local_importance_df.sort_values(by='Importance Relative', ascending=False)
top_10_features = local_importance_df.head(10)

local_shap_values = shap_values[0, :]
top_10_features['Valeur SHAP'] = [local_shap_values[random_observation_df.columns.get_loc(feature)] for feature in top_10_features['Feature']]
top_10_features['Direction'] = ['+' if value > 0 else '-' for value in top_10_features['Valeur SHAP']]
top_10_features['Importance Relative'] = top_10_features['Importance Relative']
top_10_features = top_10_features[['Feature', 'Importance Relative', 'Direction', 'Valeur SHAP']]
top_10_features['Importance Relative'] = top_10_features['Importance Relative'].apply(lambda x: f'{x:.2f}')
top_10_features['Valeur SHAP'] = top_10_features['Valeur SHAP'].apply(lambda x: f'{x:.2f}')

st.markdown("### Top 10 des Features par Importance Relative")
st.table(top_10_features)

selected_feature = st.selectbox('Selectionnez une 1ere feature :', top_20_features)
selected_feature2 = st.selectbox('Selectionnez une 2eme feature :', top_20_features)

if selected_feature:
    feature_data = X_test[selected_feature]
    fig_client_value, ax_client_value = plt.subplots()
    feature_values = feature_data.values
    client_value = random_observation_df[selected_feature].values[0]
    ax_client_value.hist(feature_values, alpha=0.5, label='Distribution')
    ax_client_value.axvline(x=client_value, color='r', linestyle='--', label='Position du client')
    plt.xlabel(selected_feature)
    plt.ylabel('Count')
    plt.title(f'Positionnement de la valeur du client pour {selected_feature}')
    plt.legend()
    st.pyplot(fig_client_value)

if selected_feature2:
    feature_data2 = X_test[selected_feature2]
    fig_client_value2, ax_client_value2 = plt.subplots()
    feature_values2 = feature_data2.values
    client_value2 = random_observation_df[selected_feature2].values[0]
    ax_client_value2.hist(feature_values2, alpha=0.5, label='Distribution')
    ax_client_value2.axvline(x=client_value2, color='r', linestyle='--', label='Position du client')
    plt.xlabel(selected_feature2)
    plt.ylabel('Count')
    plt.title(f'Positionnement de la valeur du client pour {selected_feature2}')
    plt.legend()
    st.pyplot(fig_client_value2)

if selected_feature and selected_feature2:
    accepted_clients = sub_X_test[modeleP17.predict(sub_X_test) < threshold]
    rejected_clients = sub_X_test[modeleP17.predict(sub_X_test) >= threshold]
    
    fig_bivariate, ax_bivariate = plt.subplots()
    ax_bivariate.scatter(accepted_clients[selected_feature], accepted_clients[selected_feature2], color='green', alpha=0.5, label='Credit accepte')
    ax_bivariate.scatter(rejected_clients[selected_feature], rejected_clients[selected_feature2], color='red', alpha=0.5, label='Credit non accepte')
    ax_bivariate.scatter(random_observation_df[selected_feature], random_observation_df[selected_feature2], color='blue', marker='*', s=200, label='Client selectionne')
    plt.xlabel(selected_feature)
    plt.ylabel(selected_feature2)
    plt.title(f'Analyse bi-variee entre {selected_feature} et {selected_feature2}')
    plt.legend()
    st.pyplot(fig_bivariate)