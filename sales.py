import streamlit as st
import pandas as pd
import numpy as np
import math as mt

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#import plotly.express as px

st.set_page_config(
    page_title="My Cig 90 days sales Prediction",
    initial_sidebar_state="expanded",
    menu_items={
         'Get Help': 'https://mycigmaroc.com/',
         'About': "This is app helps you estimate the Quantities to order"
    }
)


st.write(
    """
# My Cig 90 days sales Prediction
Upload the order Csv file for predictions.
"""
)

uploaded_file_tr = st.file_uploader("Upload training CSV", type=".csv")
uploaded_file = st.file_uploader("Upload CSV", type=".csv")
# ab_default = None
# result_default = None


def preprocessing(geek_60j):
    geek_60j['Vendus 15j'] = geek_60j['15 j Rabat Kbibat'] + geek_60j['15 j Mycig Meknes'] + geek_60j['15 j Casa Bourgogne'] + geek_60j[
        '15 j Rabat Hay Riad'] + geek_60j['15 j Fes'] + geek_60j['15 j Casa Idriss-1er'] + geek_60j['15 j B2B'] + geek_60j[
                             '15 j Mega mall'] + geek_60j['15 j Agadir'] + geek_60j['15 j Ecommerce'] + geek_60j['15 j Marrakech']
    geek_60j['Vendus 30j'] = geek_60j['30 j Rabat Kbibat'] + geek_60j['30 j Mycig Meknes'] + geek_60j['30 j Casa Bourgogne'] + geek_60j[
        '30 j Rabat Hay Riad'] + geek_60j['30 j Fes'] + geek_60j['30 j Casa Idriss-1er'] + geek_60j['30 j B2B'] + geek_60j[
                             '30 j Mega mall'] + geek_60j['30 j Agadir'] + geek_60j['30 j Ecommerce'] + geek_60j['30 j Marrakech']
    geek_60j['Vendus 60j'] = geek_60j['60 j Rabat Kbibat'] + geek_60j['60 j Mycig Meknes'] + geek_60j['60 j Casa Bourgogne'] + geek_60j[
        '60 j Rabat Hay Riad'] + geek_60j['60 j Fes'] + geek_60j['60 j Casa Idriss-1er'] + geek_60j['60 j B2B'] + geek_60j[
                             '60 j Mega mall'] + geek_60j['60 j Agadir'] + geek_60j['60 j Ecommerce'] + geek_60j['60 j Marrakech']
    geek_60j = geek_60j[
        ['id', 'id decl', 'Produit', 'Déclinaison', 'Catégorie', 'Stock', 'Trans. en cours', 'Stock Rabat Kbibat',
         'Alerte Rabat Kbibat', 'Ecart Rabat Kbibat', 'Stock Mycig Meknes', 'Alerte Mycig Meknes', 'Ecart Mycig Meknes',
         'Stock Casa Bourgogne', 'Alerte Casa Bourgogne', 'Ecart Casa Bourgogne', 'Stock Rabat Hay Riad',
         'Alerte Rabat Hay Riad', 'Ecart Rabat Hay Riad', 'Stock Fes', 'Alerte Fes', 'Ecart Fes',
         'Stock Casa Idriss-1er', 'Alerte Casa Idriss-1er', 'Ecart Casa Idriss-1er', 'Stock Stock central',
         'Alerte Stock central', 'Ecart Stock central', 'Stock B2B', 'Alerte B2B', 'Ecart B2B', 'Stock Mega mall',
         'Alerte Mega mall', 'Ecart Mega mall', 'Stock Agadir', 'Alerte Agadir', 'Ecart Agadir', 'Stock Ecommerce',
         'Alerte Ecommerce', 'Ecart Ecommerce', 'Stock Marrakech', 'Alerte Marrakech', 'Ecart Marrakech', 'Vendus 15j',
         'Vendus 30j', 'Vendus 60j']]
    ohe = OneHotEncoder()
    geek_60j['Déclinaison'] = geek_60j['Déclinaison'].fillna("None")
    ####### encoded categorie ##########
    encoded_cat = ohe.fit_transform(geek_60j[['Catégorie']]).toarray()
    encoded_cat_labels = np.array(ohe.categories_).ravel()
    cat_ohe = pd.DataFrame(encoded_cat, columns=encoded_cat_labels)
    ####################################
    ####### encoded declinaison ########
    encoded_dec = ohe.fit_transform(geek_60j[['Déclinaison']]).toarray()
    encoded_dec_labels = np.array(ohe.categories_).ravel()
    dec_ohe = pd.DataFrame(encoded_dec, columns=encoded_dec_labels)
    ####################################
    ######## Concat our encoded data ###
    geek_60j = pd.concat([geek_60j, dec_ohe], axis=1)
    geek_60j.drop('Déclinaison', inplace=True, axis=1)
    geek_60j = pd.concat([geek_60j, cat_ohe], axis=1)
    geek_60j.drop('Catégorie', inplace=True, axis=1)
    ####################################
    ######### Create other Declinaison ########
    geek_60j['other_dec'] = geek_60j['05'] + geek_60j['0,7ohms'] + geek_60j['0,25ohms'] + geek_60j['Pods'] + geek_60j[
        'Orange'] + geek_60j['Rose'] + geek_60j['Vert'] + geek_60j['Rainbow'] + geek_60j['1,4ohms'] + geek_60j['Bleu'] + \
                            geek_60j['Scarlet-Rouge'] + geek_60j['0,4ohms'] + geek_60j['Gold'] + geek_60j['04'] + \
                            geek_60j['Grey-Gris fonce'] + geek_60j['Violet-Neon-Chestnut-Sky M'] + geek_60j['0,6ohms'] + \
                            geek_60j['065'] + geek_60j['03'] + geek_60j['Chrome'] + geek_60j['None'] + geek_60j['Noir']
    geek_60j = geek_60j.drop(
        ['05', '0,7ohms', '0,25ohms', 'Pods', 'Orange', 'Rose', 'Vert', 'Rainbow', 'Bleu', 'Scarlet-Rouge', '0,4ohms',
         'Gold', '04', 'Grey-Gris fonce', 'Violet-Neon-Chestnut-Sky M', '0,6ohms', '065', '03', 'Chrome', 'None',
         'Noir'], axis=1)
    ######## Create other categories #########
    geek_60j['other_cat'] = geek_60j['Cartouches vides pour pods'] + geek_60j['Mods'] + geek_60j['RBA'] + geek_60j[
        'Drip tips'] + geek_60j['Adaptateurs 510'] + geek_60j['Pyrex'] + geek_60j['Autres'] + geek_60j['RTA'] + \
                            geek_60j['Clearos'] + geek_60j['Kits'] + geek_60j['Fil Résistif'] + geek_60j[
                                'RDA (Drippers)'] + geek_60j['Mods electroniques']
    geek_60j = geek_60j.drop(
        ['Cartouches vides pour pods', 'Mods', 'RBA', 'Drip tips', 'Adaptateurs 510', 'Pyrex', 'Autres', 'RTA',
         'Clearos', 'Kits', 'Fil Résistif', 'RDA (Drippers)', 'Mods electroniques'], axis=1)
    ######## Drop Id and ID Decl ##########
    geek_60j.drop(['id', 'id decl'], inplace=True, axis=1)
    geek_60j['other_cat'] = geek_60j['other_cat'] + geek_60j['Fibre & Coton']
    geek_60j['other_dec'] = geek_60j['other_dec'] + geek_60j['1,4ohms']
    geek_60j.drop(['Fibre & Coton'], inplace=True, axis=1)
    geek_60j.drop(['1,4ohms'], inplace=True, axis=1)
    geek_60j.drop(['Produit'], inplace=True, axis=1)
    return geek_60j


if uploaded_file:
    df = pd.read_csv(uploaded_file_tr)
    # The new_df got no quantities variable
    new_df = pd.read_csv(uploaded_file)
    name = new_df['Produit']
    decl = new_df['Déclinaison']
    cat = new_df['Catégorie']
    # ######### Data Preprocessing ##########
    GBR = GradientBoostingRegressor(max_depth=1, n_estimators=750, max_features='auto')
    # ########### Data preview ##############
    st.markdown("### Data preview")
    # st.dataframe(new_df)
    st.dataframe(df)

    # ohe = OneHotEncoder()
    df = preprocessing(df)
    # st.write(len(df.columns))
    new_geek = preprocessing(new_df)
    # st.dataframe(new_geek)

    st.markdown("### Predict")
    # st.dataframe(new_geek.head(10))
    if st.button('Predict'):
        y = df[["Vendus 60j"]]
        x = df.drop(columns=['Vendus 60j'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
        GBR.fit(x_train, y_train)
        # st.dataframe(new_geek)
        # st.write(len(new_geek.columns))
        get_90j = new_geek.drop(columns=['Vendus 15j'])
        #get_90j = pd.concat([get_90j, geek_60j['Vendus 60j']], axis=1)
        pred = GBR.predict(get_90j)
        pred = np.ceil(pred)
        pred_90j = pd.DataFrame(abs(pred), columns=['Predicted V90J'])
        ven_90j = pred_90j['Predicted V90J'] + get_90j['Vendus 60j']
        # , pd.DataFrame(abs(pred_90j), columns=['Predicted V90J'])
        new_prediction = pd.concat(
            [name, cat, decl, get_90j['Vendus 60j'], pd.DataFrame(ven_90j, columns=['Vendus 90j']), get_90j['Stock']], axis=1)
        st.dataframe(new_prediction)
        # res = pd.concat([name, cat, decl, pd.DataFrame(pred, columns=['Predicted V90J'])], axis=1)
        # st.write(res)
        # y7_pred = GBR.predict(x_test)
        # mae = mean_absolute_error(y_test, y7_pred)
        # mse = mean_squared_error(y_test, y7_pred)
        # rmse = np.sqrt(mse)
        # r2 = r2_score(y_test, y7_pred)
        # st.write(mae)
        # st.write(rmse)
        # st.write(r2)
# fig = px.pie(res, values='Predicted', names='Catégorie', title='Population of European continent')
# fig.show()
