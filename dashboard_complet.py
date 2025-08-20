import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
from math import exp, sqrt

# =============================================================================
# 1. CONFIGURATION DE LA PAGE ET DES LANGUES
# =============================================================================
st.set_page_config(layout="wide", page_title="Analyse Stratégique H₂ Vert")

# Dictionnaire de traductions mis à jour avec Anglais et Italien
translations = {
    "fr": {
        "page_title": "Analyse Approfondie de Projet d'Hydrogène Vert", "sidebar_title": "Paramètres de Simulation",
        "kpi_header": "Synthèse Exécutive", "npv_metric": "VAN Statique", "irr_metric": "TRI (IRR)",
        "analysis_header": "Analyse Financière & Environnementale",
        "risk_header": "Analyse de Risque et de Sensibilité",
        "assumptions_header": "Hypothèses et Méthodologie du Modèle", "financial_assumptions": "Hypothèses Financières & Techniques", "capex_assumptions": "Détail du Capital Expenditure (CAPEX)",
        "opex_assumptions": "Détail de l'Operational Expenditure (OPEX)", "methodology_header": "Formules de Calcul Clés",
        "parameter": "Paramètre", "value": "Valeur", "h2_sale_price": "Prix de Vente H₂ (€/kg)", "discount_rate": "Taux d'Actualisation (%)",
        "inflation_rate": "Taux d'Inflation (%)", "project_lifetime": "Durée de vie du projet (ans)", "tax_rate": "Taux d'imposition (%)",
        "total_initial_capex": "CAPEX Initial Total (€)", "grant_percent": "Subvention (%)", "adjusted_capex": "CAPEX Ajusté (€)",
        "maintenance": "Maintenance", "insurance": "Assurance", "personnel": "Personnel", "consumables": "Consommables", "water": "Eau", "total_fixed_opex": "Total OPEX Fixe de Base (€)",
        "elec_price": "Prix de l'Électricité (€/MWh)", "lcoh_metric": "LCOH (Coût de Production)", "payback_period": "Période de Retour", "tab_cost_breakdown": "Décomposition des Coûts",
        "tab_opex_evolution": "Évolution des Coûts Annuels", "tab_cumulative_cashflow": "Analyse de la Rentabilité", "tab_cashflow_table": "Tableau des Flux de Trésorerie",
        "tab_tornado": "Sensibilité à 1 Variable (Tornado)", "tab_heatmap": "Sensibilité à 2 Variables (Heatmap)", "tab_monte_carlo": "Simulation de Risque (Monte Carlo)",
        "cost_breakdown_title": "Décomposition des Coûts Totals Actualisés", "capex_label": "CAPEX Ajusté", "fixed_opex_label": "OPEX Fixe", "variable_opex_label": "OPEX Variable (Électricité)",
        "replacement_cost_label": "Coûts de Remplacement",
        "opex_evolution_title": "Évolution Annuelle de l'OPEX et des Remplacements", "cumulative_cashflow_chart_title": "Flux de Trésorerie Cumulés et Période de Retour",
        "mc_button": "Lancer la Simulation de Monte Carlo", "mc_title": "Distribution de la VAN (5,000 Scénarios)", "mc_prob_success": "Probabilité de rentabilité (VAN > 0)",
        "mc_explanation": "La simulation de Monte Carlo teste des milliers de scénarios en faisant varier les paramètres clés (prix H₂, prix électricité, CAPEX) selon des distributions de probabilité pour évaluer l'éventail des résultats possibles et le risque global du projet.",
        "capex_sidebar_header": "Détail du CAPEX Modifiable (€)", "capex_design": "Design, Sécurité, Management", "capex_pv": "Centrale PV",
        "capex_battery": "Système de Batterie (BESS)", "capex_electrolyzer": "Chaîne Électrolyseur", "capex_auxiliary": "Systèmes Auxiliaires", "capex_civil": "Génie Civil",
        "environmental_header": "Analyse Environnementale", "co2_avoided_metric": "Émissions de CO₂ Évitées",
        "co2_sidebar_header": "Paramètres Environnementaux", "grid_emission_factor": "Facteur d'Émission du Réseau (tCO₂/MWh)",
        "h2_energy_density": "Densité Énergétique H₂ (LHV, kWh/kg)", "co2_chart_title": "Émissions de CO₂ Évitées Cumulées",
        "replacement_sidebar_header": "Coûts de Remplacement", "replacement_year_stack": "Année de rempl. (Stack)", "replacement_cost_stack": "Coût de rempl. (Stack) (€)",
        "replacement_year_bess": "Année de rempl. (Batterie)", "replacement_cost_bess": "Coût de rempl. (Batterie) (€)",
        "mc_pie_title": "Synthèse des Scénarios", "mc_pie_profitable": "Rentables", "mc_pie_unprofitable": "Non Rentables",
        "real_options_header": "Valorisation Stratégique (Options Réelles)",
        "real_options_desc": "Cette section évalue la valeur de la flexibilité managériale. Une VAN statique négative peut être compensée par la valeur d'une option stratégique future, comme celle d'agrandir le projet si le marché devient favorable.",
        "option_value_metric": "Valeur de l'Option d'Expansion",
        "total_value_metric": "Valeur Stratégique Totale (VAN + Option)",
        "option_sidebar_header": "Paramètres de l'Option d'Expansion",
        "option_years": "Horizon de décision (années)",
        "option_volatility": "Volatilité des flux de trésorerie (%)",
        "option_expansion_cost": "Coût de l'expansion (€)",
        "option_expansion_factor": "Facteur d'augmentation des revenus (%)",
        "option_help": "Valeur de la flexibilité pour agrandir le projet si les conditions de marché sont favorables.",
        "total_value_help": "La véritable valeur du projet, incluant son potentiel stratégique futur.",
        "tornado_title": "Analyse d'Impact des Variables sur la VAN (Tornado)",
        "tornado_button": "Lancer l'Analyse de Sensibilité",
        "heatmap_title": "Impact Combiné de Deux Variables sur la VAN",
        "heatmap_var1": "Choisissez la première variable (Axe X)",
        "heatmap_var2": "Choisissez la deuxième variable (Axe Y)",
        "heatmap_button": "Générer la Heatmap",
    },
    "en": {
        "page_title": "In-Depth Green Hydrogen Project Analysis", "sidebar_title": "Simulation Parameters",
        "kpi_header": "Executive Summary", "npv_metric": "Static NPV", "irr_metric": "IRR",
        "analysis_header": "Financial & Environmental Analysis",
        "risk_header": "Risk and Sensitivity Analysis",
        "assumptions_header": "Model Assumptions and Methodology", "financial_assumptions": "Financial & Technical Assumptions", "capex_assumptions": "Capital Expenditure (CAPEX) Detail",
        "opex_assumptions": "Operational Expenditure (OPEX) Detail", "methodology_header": "Key Calculation Formulas",
        "parameter": "Parameter", "value": "Value", "h2_sale_price": "H₂ Sale Price (€/kg)", "discount_rate": "Discount Rate (%)",
        "inflation_rate": "Inflation Rate (%)", "project_lifetime": "Project Lifetime (years)", "tax_rate": "Tax Rate (%)",
        "total_initial_capex": "Total Initial CAPEX (€)", "grant_percent": "Grant (%)", "adjusted_capex": "Adjusted CAPEX (€)",
        "maintenance": "Maintenance", "insurance": "Insurance", "personnel": "Personnel", "consumables": "Consumables", "water": "Water", "total_fixed_opex": "Total Base Fixed OPEX (€)",
        "elec_price": "Electricity Price (€/MWh)", "lcoh_metric": "LCOH (Production Cost)", "payback_period": "Payback Period", "tab_cost_breakdown": "Cost Breakdown",
        "tab_opex_evolution": "Annual Cost Evolution", "tab_cumulative_cashflow": "Profitability Analysis", "tab_cashflow_table": "Cash Flow Table",
        "tab_tornado": "1-Variable Sensitivity (Tornado)", "tab_heatmap": "2-Variable Sensitivity (Heatmap)", "tab_monte_carlo": "Risk Simulation (Monte Carlo)",
        "cost_breakdown_title": "Breakdown of Total Discounted Costs", "capex_label": "Adjusted CAPEX", "fixed_opex_label": "Fixed OPEX", "variable_opex_label": "Variable OPEX (Electricity)",
        "replacement_cost_label": "Replacement Costs",
        "opex_evolution_title": "Annual OPEX and Replacement Evolution", "cumulative_cashflow_chart_title": "Cumulative Cash Flow and Payback Period",
        "mc_button": "Run Monte Carlo Simulation", "mc_title": "NPV Distribution (5,000 Scenarios)", "mc_prob_success": "Probability of Profitability (NPV > 0)",
        "mc_explanation": "The Monte Carlo simulation tests thousands of scenarios by varying key parameters (H₂ price, electricity price, CAPEX) according to probability distributions to assess the range of possible outcomes and the overall project risk.",
        "capex_sidebar_header": "Editable CAPEX Detail (€)", "capex_design": "Design, Safety, Management", "capex_pv": "PV Plant",
        "capex_battery": "Battery System (BESS)", "capex_electrolyzer": "Electrolyzer Chain", "capex_auxiliary": "Auxiliary Systems", "capex_civil": "Civil Works",
        "environmental_header": "Environmental Analysis", "co2_avoided_metric": "CO₂ Emissions Avoided",
        "co2_sidebar_header": "Environmental Parameters", "grid_emission_factor": "Grid Emission Factor (tCO₂/MWh)",
        "h2_energy_density": "H₂ Energy Density (LHV, kWh/kg)", "co2_chart_title": "Cumulative CO₂ Emissions Avoided",
        "replacement_sidebar_header": "Replacement Costs", "replacement_year_stack": "Repl. Year (Stack)", "replacement_cost_stack": "Repl. Cost (Stack) (€)",
        "replacement_year_bess": "Repl. Year (Battery)", "replacement_cost_bess": "Repl. Cost (Battery) (€)",
        "mc_pie_title": "Scenario Summary", "mc_pie_profitable": "Profitable", "mc_pie_unprofitable": "Unprofitable",
        "real_options_header": "Strategic Valuation (Real Options)",
        "real_options_desc": "This section evaluates the value of managerial flexibility. A negative static NPV can be offset by the value of a future strategic option, such as expanding the project if the market becomes favorable.",
        "option_value_metric": "Value of Expansion Option",
        "total_value_metric": "Total Strategic Value (NPV + Option)",
        "option_sidebar_header": "Expansion Option Parameters",
        "option_years": "Decision Horizon (years)",
        "option_volatility": "Cash Flow Volatility (%)",
        "option_expansion_cost": "Expansion Cost (€)",
        "option_expansion_factor": "Revenue Increase Factor (%)",
        "option_help": "Value of the flexibility to expand the project if market conditions are favorable.",
        "total_value_help": "The true value of the project, including its future strategic potential.",
        "tornado_title": "Impact Analysis of Variables on NPV (Tornado)",
        "tornado_button": "Run Sensitivity Analysis",
        "heatmap_title": "Combined Impact of Two Variables on NPV",
        "heatmap_var1": "Choose the first variable (X-Axis)",
        "heatmap_var2": "Choose the second variable (Y-Axis)",
        "heatmap_button": "Generate Heatmap",
    },
    "it": {
        "page_title": "Analisi Approfondita del Progetto Idrogeno Verde", "sidebar_title": "Parametri di Simulazione",
        "kpi_header": "Sintesi Esecutiva", "npv_metric": "VAN Statico", "irr_metric": "TIR (IRR)",
        "analysis_header": "Analisi Finanziaria e Ambientale",
        "risk_header": "Analisi del Rischio e di Sensibilità",
        "assumptions_header": "Ipotesi e Metodologia del Modello", "financial_assumptions": "Ipotesi Finanziarie e Tecniche", "capex_assumptions": "Dettaglio Capital Expenditure (CAPEX)",
        "opex_assumptions": "Dettaglio Operational Expenditure (OPEX)", "methodology_header": "Formule di Calcolo Chiave",
        "parameter": "Parametro", "value": "Valore", "h2_sale_price": "Prezzo di Vendita H₂ (€/kg)", "discount_rate": "Tasso di Attualizzazione (%)",
        "inflation_rate": "Tasso di Inflazione (%)", "project_lifetime": "Durata del progetto (anni)", "tax_rate": "Aliquota Fiscale (%)",
        "total_initial_capex": "CAPEX Iniziale Totale (€)", "grant_percent": "Sovvenzione (%)", "adjusted_capex": "CAPEX Rettificato (€)",
        "maintenance": "Manutenzione", "insurance": "Assicurazione", "personnel": "Personale", "consumables": "Consumabili", "water": "Acqua", "total_fixed_opex": "Totale OPEX Fisso di Base (€)",
        "elec_price": "Prezzo Elettricità (€/MWh)", "lcoh_metric": "LCOH (Costo di Produzione)", "payback_period": "Periodo di Ritorno", "tab_cost_breakdown": "Scomposizione dei Costi",
        "tab_opex_evolution": "Evoluzione dei Costi Annuali", "tab_cumulative_cashflow": "Analisi di Redditività", "tab_cashflow_table": "Tabella Flussi di Cassa",
        "tab_tornado": "Sensibilità a 1 Variabile (Tornado)", "tab_heatmap": "Sensibilità a 2 Variabili (Heatmap)", "tab_monte_carlo": "Simulazione del Rischio (Monte Carlo)",
        "cost_breakdown_title": "Scomposizione dei Costi Totali Attualizzati", "capex_label": "CAPEX Rettificato", "fixed_opex_label": "OPEX Fisso", "variable_opex_label": "OPEX Variabile (Elettricità)",
        "replacement_cost_label": "Costi di Sostituzione",
        "opex_evolution_title": "Evoluzione Annuale OPEX e Sostituzioni", "cumulative_cashflow_chart_title": "Flusso di Cassa Cumulato e Periodo di Ritorno",
        "mc_button": "Esegui Simulazione Monte Carlo", "mc_title": "Distribuzione del VAN (5.000 Scenari)", "mc_prob_success": "Probabilità di Redditività (VAN > 0)",
        "mc_explanation": "La simulazione Monte Carlo testa migliaia di scenari variando i parametri chiave (prezzo H₂, prezzo elettricità, CAPEX) secondo distribuzioni di probabilità per valutare la gamma di risultati possibili e il rischio complessivo del progetto.",
        "capex_sidebar_header": "Dettaglio CAPEX Modificabile (€)", "capex_design": "Progettazione, Sicurezza, Gestione", "capex_pv": "Impianto FV",
        "capex_battery": "Sistema di Batterie (BESS)", "capex_electrolyzer": "Catena Elettrolizzatore", "capex_auxiliary": "Sistemi Ausiliari", "capex_civil": "Opere Civili",
        "environmental_header": "Analisi Ambientale", "co2_avoided_metric": "Emissioni di CO₂ Evitate",
        "co2_sidebar_header": "Parametri Ambientali", "grid_emission_factor": "Fattore di Emissione della Rete (tCO₂/MWh)",
        "h2_energy_density": "Densità Energetica H₂ (PCI, kWh/kg)", "co2_chart_title": "Emissioni di CO₂ Evitate Cumulate",
        "replacement_sidebar_header": "Costi di Sostituzione", "replacement_year_stack": "Anno sost. (Stack)", "replacement_cost_stack": "Costo sost. (Stack) (€)",
        "replacement_year_bess": "Anno sost. (Batteria)", "replacement_cost_bess": "Costo sost. (Batteria) (€)",
        "mc_pie_title": "Sintesi degli Scenari", "mc_pie_profitable": "Redditizi", "mc_pie_unprofitable": "Non Redditizi",
        "real_options_header": "Valutazione Strategica (Opzioni Reali)",
        "real_options_desc": "Questa sezione valuta il valore della flessibilità manageriale. Un VAN statico negativo può essere compensato dal valore di un'opzione strategica futura, come quella di espandere il progetto se il mercato diventa favorevole.",
        "option_value_metric": "Valore dell'Opzione di Espansione",
        "total_value_metric": "Valore Strategico Totale (VAN + Opzione)",
        "option_sidebar_header": "Parametri dell'Opzione di Espansione",
        "option_years": "Orizzonte decisionale (anni)",
        "option_volatility": "Volatilità dei flussi di cassa (%)",
        "option_expansion_cost": "Costo dell'espansione (€)",
        "option_expansion_factor": "Fattore di aumento dei ricavi (%)",
        "option_help": "Valore della flessibilità di espandere il progetto se le condizioni di mercato sono favorevoli.",
        "total_value_help": "Il vero valore del progetto, incluso il suo potenziale strategico futuro.",
        "tornado_title": "Analisi d'Impatto delle Variabili sul VAN (Tornado)",
        "tornado_button": "Esegui Analisi di Sensibilità",
        "heatmap_title": "Impatto Combinato di Due Variabili sul VAN",
        "heatmap_var1": "Scegli la prima variabile (Asse X)",
        "heatmap_var2": "Scegli la seconda variabile (Asse Y)",
        "heatmap_button": "Genera Heatmap",
    }
}

# =============================================================================
# 2. FONCTIONS DE CALCUL (LE MOTEUR DE L'ANALYSE)
# =============================================================================
@st.cache_data(show_spinner=False)
def generate_cashflows(params):
    try:
        # Assurez-vous que le fichier Excel est accessible par le script
        df_engine_base = pd.read_excel(params["excel_file"], sheet_name='LCOH_Engine')
    except FileNotFoundError:
        st.error(f"Error: The file '{params['excel_file']}' was not found. Please make sure it's in the same directory as the script.")
        return pd.DataFrame(), {}
    except Exception as e:
        st.error(f"Error reading Excel file. Details: {e}")
        return pd.DataFrame(), {}
    
    years = np.arange(params["project_lifetime"] + 1)
    cashflows = pd.DataFrame(index=years)
    
    # OPEX
    fixed_opex_inflated = [params["base_total_fixed_opex"] * (1 + params["inflation_rate"])**(y-1) for y in range(1, params["project_lifetime"] + 1)]
    variable_opex_by_year = df_engine_base.loc[1:, 'Grid Draw (MWh)'].head(params["project_lifetime"]).values * params["elec_price_mwh"]
    cashflows['Fixed_OPEX'] = 0.0; cashflows.loc[1:, 'Fixed_OPEX'] = fixed_opex_inflated
    cashflows['Variable_OPEX'] = 0.0; cashflows.loc[1:, 'Variable_OPEX'] = variable_opex_by_year
    cashflows['Total OPEX'] = 0.0; cashflows.loc[1:, 'Total OPEX'] = np.array(fixed_opex_inflated) + np.array(variable_opex_by_year)
    
    # Revenus et Calculs financiers
    cashflows['Revenue'] = 0.0; cashflows.loc[1:, 'Revenue'] = params["annual_h2_production_kg"] * params["h2_sale_price"]
    cashflows['Depreciation'] = 0.0; cashflows.loc[1:, 'Depreciation'] = params["total_initial_capex"] / params["project_lifetime"]
    
    cashflows['Replacement_CAPEX'] = 0.0
    if params['replacement_year_stack'] <= params['project_lifetime']:
        cashflows.loc[params['replacement_year_stack'], 'Replacement_CAPEX'] += params['replacement_cost_stack']
    if params['replacement_year_bess'] <= params['project_lifetime']:
        cashflows.loc[params['replacement_year_bess'], 'Replacement_CAPEX'] += params['replacement_cost_bess']

    cashflows['EBT'] = cashflows['Revenue'] - cashflows['Total OPEX'] - cashflows['Depreciation']
    cashflows['Taxes'] = cashflows['EBT'].apply(lambda x: x * params["tax_rate"] if x > 0 else 0)
    cashflows['Net Earnings'] = cashflows['EBT'] - cashflows['Taxes']
    cashflows['Free Cash Flow'] = cashflows['Net Earnings'] + cashflows['Depreciation']
    cashflows['Full Cash Flow'] = cashflows['Free Cash Flow'] - cashflows['Replacement_CAPEX']
    cashflows.loc[0, 'Full Cash Flow'] = -params["adjusted_capex"]
    cashflows['Cumulative Cash Flow'] = cashflows['Full Cash Flow'].cumsum()
    
    # Calculs des KPIs
    discount_factors = np.array([(1 / (1 + params["discount_rate"]) ** year) for year in years])
    cashflows['Discounted FCF'] = cashflows['Full Cash Flow'] * discount_factors
    npv = np.sum(cashflows['Discounted FCF'])
    
    try:
        irr = npf.irr(cashflows['Full Cash Flow'])
    except:
        irr = -1 # Valeur d'erreur si le calcul échoue

    working_capital = cashflows['Total OPEX'].iloc[1] * params.get('working_capital_percent', 0.15)
    
    # *** CORRECTION APPLIQUÉE ICI ***
    # Utilisation de la production totale simple (non-actualisée) pour le LCOH
    total_h2_production_simple = params["annual_h2_production_kg"] * params["project_lifetime"]
    
    discounted_opex = np.sum(cashflows.loc[1:, 'Total OPEX'] * discount_factors[1:])
    decomm_cost = params["total_initial_capex"] * 0.00
    discounted_decomm = decomm_cost / (1 + params["discount_rate"])**params["project_lifetime"]
    discounted_replacement = np.sum(cashflows.loc[1:, 'Replacement_CAPEX'] * discount_factors[1:])
    
    # *** CORRECTION APPLIQUÉE ICI ***
    # La formule utilise maintenant la production totale simple
    if total_h2_production_simple > 0:
        lcoh = (params["adjusted_capex"] + discounted_opex + discounted_decomm + working_capital + discounted_replacement) / total_h2_production_simple
    else:
        lcoh = 0

    try:
        payback_year_val = cashflows[cashflows['Cumulative Cash Flow'] > 0].index[0]
        payback_year = f"{payback_year_val} years"
    except IndexError:
        payback_year_val, payback_year = np.inf, f"> {params['project_lifetime']} years"
        
    # Calculs environnementaux
    energy_produced_mwh = (params["annual_h2_production_kg"] * params["h2_energy_density"]) / 1000
    annual_co2_avoided = energy_produced_mwh * params["grid_emission_factor"]
    cashflows['CO2_Avoided_Tonnes'] = 0.0; cashflows.loc[1:, 'CO2_Avoided_Tonnes'] = annual_co2_avoided
    cashflows['Cumulative_CO2_Avoided'] = cashflows['CO2_Avoided_Tonnes'].cumsum()

    pv_of_future_cashflows = np.sum(cashflows.loc[1:, 'Discounted FCF'])
    
    kpi = {'npv': npv, 'irr': irr, 'lcoh': lcoh, 'payback': payback_year, 'payback_val': payback_year_val, 'total_co2_avoided': cashflows['Cumulative_CO2_Avoided'].iloc[-1], 'pv_future_fcf': pv_of_future_cashflows}
    return cashflows, kpi

@st.cache_data(show_spinner=False)
def calculate_real_option(params, kpi):
    S=kpi['pv_future_fcf']; K=params['option_expansion_cost']; T_opt=params['option_years']; r=params['discount_rate']; sigma=params['option_volatility']
    if sigma == 0 or T_opt == 0 or S <= 0: return 0.0
    # Modèle binomial simple pour l'évaluation de l'option
    u = exp(sigma * sqrt(T_opt)); d = 1/u
    if not (d < exp(r * T_opt) < u): return 0.0 # Condition de non-arbitrage
    p = (exp(r * T_opt) - d) / (u - d)
    option_value_up = max(0, S * u * (1 + params['option_expansion_factor']) - K)
    option_value_down = max(0, S * d * (1 + params['option_expansion_factor']) - K)
    option_price = exp(-r * T_opt) * (p * option_value_up + (1-p) * option_value_down)
    return option_price

# =============================================================================
# 3. FONCTIONS GRAPHIQUES ET D'ANALYSE AVANCÉE
# =============================================================================

def plot_cost_breakdown_bar(cashflows, params, T):
    discount_factors = np.array([(1 / (1 + params["discount_rate"]) ** year) for year in range(params["project_lifetime"] + 1)])
    disc_fixed_opex = np.sum(cashflows.loc[1:, 'Fixed_OPEX'] * discount_factors[1:])
    disc_variable_opex = np.sum(cashflows.loc[1:, 'Variable_OPEX'] * discount_factors[1:])
    disc_replacement = np.sum(cashflows.loc[1:, 'Replacement_CAPEX'] * discount_factors[1:])
    labels = [T['capex_label'], T['fixed_opex_label'], T['variable_opex_label'], T['replacement_cost_label']]
    values = [params['adjusted_capex'], disc_fixed_opex, disc_variable_opex, disc_replacement]
    total_cost = sum(values)
    percentages_text = [f"{val/total_cost:.1%}" for val in values] if total_cost > 0 else ["0.0%"]*4
    fig = go.Figure(go.Bar(y=labels, x=values, text=percentages_text, textposition='inside', insidetextanchor='middle', orientation='h', marker_color=['#4C78A8', '#F58518', '#E45756', '#72B7B2']))
    fig.update_layout(title_text=T['cost_breakdown_title'], xaxis_title="Discounted Cost (€)", template="plotly_dark", height=400, margin=dict(l=200, t=50))
    fig.update_yaxes(autorange="reversed")
    return fig

def plot_opex_evolution(df, T):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[1:], y=df.loc[1:, 'Fixed_OPEX'], mode='lines', stackgroup='one', name=T['fixed_opex_label'], line=dict(width=0.5, color='#F58518')))
    fig.add_trace(go.Scatter(x=df.index[1:], y=df.loc[1:, 'Variable_OPEX'], mode='lines', stackgroup='one', name=T['variable_opex_label'], line=dict(width=0.5, color='#E45756')))
    replacements = df[df['Replacement_CAPEX'] > 0]
    if not replacements.empty:
        fig.add_trace(go.Bar(x=replacements.index, y=replacements['Replacement_CAPEX'], name=T['replacement_cost_label'], marker_color='#4C78A8'))
    fig.update_layout(title_text=T['opex_evolution_title'], xaxis_title="Year", yaxis_title="Annual Cost (€)", template="plotly_dark", height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def plot_cumulative_cashflow(df, kpi, T):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative Cash Flow'], mode='lines+markers', name='Cumulative Cash Flow'))
    fig.add_hline(y=0, line_dash="dash", line_color="white")
    if kpi['payback_val'] != np.inf:
        fig.add_vline(x=kpi['payback_val'], line_dash="dash", line_color="green", annotation_text=f"{T['payback_period']}: {kpi['payback_val']} years", annotation_position="top left")
    fig.update_layout(title_text=T['cumulative_cashflow_chart_title'], template="plotly_dark", height=400)
    return fig

def plot_co2_avoided(df, T):
    fig = go.Figure(data=[go.Bar(x=df.index[1:], y=df.loc[1:, 'Cumulative_CO2_Avoided'], marker_color='#00A0B0')])
    fig.update_layout(title_text=T['co2_chart_title'], xaxis_title="Year", yaxis_title="Tonnes of CO₂", template="plotly_dark", height=400)
    return fig

def plot_monte_carlo_pie(prob_success, T):
    prob_fail = 100 - prob_success
    labels = [T['mc_pie_profitable'], T['mc_pie_unprofitable']]
    values = [prob_success, prob_fail]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=['#54A24B', '#E45756'])])
    fig.update_layout(title_text=T['mc_pie_title'], template="plotly_dark", height=400, legend=dict(orientation="h", yanchor="bottom", y=-0.2))
    return fig

@st.cache_data(show_spinner="Calculating...")
def run_tornado_analysis(_params, T):
    base_npv = generate_cashflows(_params)[1]['npv']
    variables_to_test = {
        'h2_sale_price': {'name': T['h2_sale_price'], 'range': 0.15},
        'adjusted_capex': {'name': T['adjusted_capex'], 'range': 0.15},
        'elec_price_mwh': {'name': T['elec_price'], 'range': 0.20},
        'base_total_fixed_opex': {'name': T['total_fixed_opex'], 'range': 0.20}
    }
    results = []
    for var, details in variables_to_test.items():
        low_params = _params.copy()
        low_params[var] *= (1 - details['range'])
        npv_low = generate_cashflows(low_params)[1]['npv']
        
        high_params = _params.copy()
        high_params[var] *= (1 + details['range'])
        npv_high = generate_cashflows(high_params)[1]['npv']
        
        results.append({'variable': details['name'], 'impact': abs(npv_high - npv_low), 'npv_low': npv_low, 'npv_high': npv_high})
        
    df = pd.DataFrame(results).sort_values(by='impact', ascending=True)
    return df

def plot_tornado(df, base_npv, T):
    fig = go.Figure()
    # Base bar for low values
    fig.add_trace(go.Bar(
        y=df['variable'],
        x=df['npv_low'] - base_npv,
        base=base_npv,
        orientation='h',
        name='Low Case',
        marker_color='#E45756'
    ))
    # Bar for high values
    fig.add_trace(go.Bar(
        y=df['variable'],
        x=df['npv_high'] - base_npv,
        base=base_npv,
        orientation='h',
        name='High Case',
        marker_color='#54A24B'
    ))
    fig.add_vline(x=base_npv, line_dash="dash", line_color="white", annotation_text="Base NPV")
    fig.update_layout(
        barmode='overlay',
        title_text=T['tornado_title'],
        xaxis_title="Net Present Value (NPV) in €",
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

@st.cache_data(show_spinner="Calculating...")
def run_heatmap_analysis(_params, var1_key, var2_key, var1_range=0.2, var2_range=0.2, steps=10):
    var1_values = np.linspace(_params[var1_key] * (1-var1_range), _params[var1_key] * (1+var1_range), steps)
    var2_values = np.linspace(_params[var2_key] * (1-var2_range), _params[var2_key] * (1+var2_range), steps)
    npv_matrix = np.zeros((steps, steps))

    for i, val1 in enumerate(var1_values):
        for j, val2 in enumerate(var2_values):
            temp_params = _params.copy()
            temp_params[var1_key] = val1
            temp_params[var2_key] = val2
            npv_matrix[j, i] = generate_cashflows(temp_params)[1]['npv']
    
    return npv_matrix, var1_values, var2_values

def plot_heatmap(npv_matrix, x_values, y_values, T, var1_name, var2_name):
    fig = go.Figure(data=go.Heatmap(
        z=npv_matrix,
        x=x_values,
        y=y_values,
        colorscale='RdYlGn',
        zmid=0
    ))
    fig.update_layout(
        title=T['heatmap_title'],
        xaxis_title=var1_name,
        yaxis_title=var2_name,
        template="plotly_dark",
        height=500
    )
    return fig

@st.cache_data(show_spinner=False)
def run_monte_carlo(_params, n_simulations=5000):
    results = [generate_cashflows({**_params, "h2_sale_price": np.random.triangular(_params["h2_sale_price"]*0.8, _params["h2_sale_price"], _params["h2_sale_price"]*1.2), "elec_price_mwh": np.random.normal(_params["elec_price_mwh"], _params["elec_price_mwh"]*0.2), "adjusted_capex": np.random.normal(_params["adjusted_capex"], _params["adjusted_capex"]*0.1)})[1]['npv'] for _ in range(n_simulations)]
    return np.array(results)

# =============================================================================
# 4. INTERFACE UTILISATEUR (LE DASHBOARD)
# =============================================================================
lang_options = {"Français": "fr", "English": "en", "Italiano": "it"}
lang_selection = st.sidebar.selectbox("Language / Langue / Lingua", options=list(lang_options.keys()))
lang_code = lang_options[lang_selection]
T = translations[lang_code]

st.title(T['page_title'])
st.sidebar.title(T['sidebar_title'])

# --- SIDEBAR INPUTS ---
with st.sidebar.expander(T['capex_sidebar_header'], expanded=True):
    capex_items = {
        T['capex_design']: st.number_input(T['capex_design'], value=422000, key='dsm', format="%d"),
        T['capex_pv']: st.number_input(T['capex_pv'], value=2940000, key='pv', format="%d"),
        T['capex_battery']: st.number_input(T['capex_battery'], value=2920000, key='bess', format="%d"),
        T['capex_electrolyzer']: st.number_input(T['capex_electrolyzer'], value=1781000, key='elec', format="%d"),
        T['capex_auxiliary']: st.number_input(T['capex_auxiliary'], value=210000, key='aux', format="%d"),
        T['capex_civil']: st.number_input(T['capex_civil'], value=854000, key='civil', format="%d")
    }
    p_total_capex = sum(capex_items.values())

st.sidebar.header(T['financial_assumptions'])
p_h2_price = st.sidebar.slider(T['h2_sale_price'], 4.0, 12.0, 6.0, 0.1)
p_grant_percent = st.sidebar.slider(T['grant_percent'], 0, 100, 75)
p_discount_rate = st.sidebar.slider(T['discount_rate'], 3.0, 10.0, 5.0, 0.1)
p_inflation_rate = st.sidebar.slider(T['inflation_rate'], 1.0, 5.0, 2.0, 0.1)
st.sidebar.header(T['opex_assumptions'])
p_elec_price = st.sidebar.number_input(T['elec_price'], value=118)
st.sidebar.header(T['replacement_sidebar_header'])
p_replacement_year_stack = st.sidebar.slider(T['replacement_year_stack'], 1, 20, 10)
p_replacement_cost_stack = st.sidebar.number_input(T['replacement_cost_stack'], value=350000, format="%d")
p_replacement_year_bess = st.sidebar.slider(T['replacement_year_bess'], 1, 20, 15)
p_replacement_cost_bess = st.sidebar.number_input(T['replacement_cost_bess'], value=135000, format="%d")
st.sidebar.header(T['co2_sidebar_header'])
p_grid_emission_factor = st.sidebar.slider(T['grid_emission_factor'], 0.0, 0.5, 0.25, 0.01)
p_h2_energy_density = st.sidebar.number_input(T['h2_energy_density'], value=33.3)
st.sidebar.header(T['option_sidebar_header'])
p_option_years = st.sidebar.slider(T['option_years'], 1, 10, 5); p_option_volatility = st.sidebar.slider(T['option_volatility'], 10, 50, 30); p_option_expansion_cost = st.sidebar.number_input(T['option_expansion_cost'], value=1500000, format="%d"); p_option_expansion_factor = st.sidebar.slider(T['option_expansion_factor'], 10, 100, 50)

# --- CALCULATIONS ---
p_fixed_opex_base = {"Maintenance": p_total_capex * 0.005, "Insurance": p_total_capex * 0.0025, "Personnel": 80000, "Consommables": 15000, "Eau": 8000}
p_fixed_opex = sum(p_fixed_opex_base.values())
params = {"excel_file": "ECONOMIC TEST 4.19.xlsx", "h2_sale_price": p_h2_price, "discount_rate": p_discount_rate / 100.0, "inflation_rate": p_inflation_rate / 100.0, "total_initial_capex": p_total_capex, "grant_percent": p_grant_percent / 100.0, "adjusted_capex": p_total_capex * (1 - p_grant_percent / 100.0), "elec_price_mwh": p_elec_price, "base_total_fixed_opex": p_fixed_opex, "project_lifetime": 20, "annual_h2_production_kg": 79840, "tax_rate": 0.24, "working_capital_percent": 0.15, "replacement_year_stack": p_replacement_year_stack, "replacement_cost_stack": p_replacement_cost_stack, "replacement_year_bess": p_replacement_year_bess, "replacement_cost_bess": p_replacement_cost_bess, "grid_emission_factor": p_grid_emission_factor, "h2_energy_density": p_h2_energy_density, "option_years": p_option_years, "option_volatility": p_option_volatility / 100.0, "option_expansion_cost": p_option_expansion_cost, "option_expansion_factor": p_option_expansion_factor / 100.0}
cashflows_df, kpi_results = generate_cashflows(params)
option_value = calculate_real_option(params, kpi_results)

# --- MAIN PAGE LAYOUT ---
st.header(T['kpi_header'])
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(T['npv_metric'], f"{kpi_results['npv']:,.0f} €")
col2.metric(T['irr_metric'], f"{kpi_results['irr']:.2%}" if kpi_results['irr'] != -1 else "N/A")
col3.metric(T['lcoh_metric'], f"{kpi_results['lcoh']:.2f} €/kg")
col4.metric(T['payback_period'], kpi_results['payback'])
col5.metric(T['co2_avoided_metric'], f"{kpi_results['total_co2_avoided']:,.0f} tCO₂")

with st.expander(T['assumptions_header']):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader(T['financial_assumptions']); st.table(pd.DataFrame({T['parameter']: [T['discount_rate'], T['inflation_rate'], T['tax_rate'], T['project_lifetime']], T['value']: [f"{params['discount_rate']*100:.1f}%", f"{params['inflation_rate']*100:.1f}%", f"{params['tax_rate']*100:.0f}%", f"{params['project_lifetime']} years"]}))
    with col_b:
        st.subheader(T['capex_assumptions']); st.table(pd.DataFrame(capex_items.items(), columns=[T['parameter'], T['value']]).style.format({T['value']: "€{:,.0f}"}))
    with col_c:
        st.subheader(T['opex_assumptions']); st.table(pd.DataFrame(p_fixed_opex_base.items(), columns=[T['parameter'], T['value']]).style.format({T['value']: "€{:,.0f}"}))
    st.subheader(T['methodology_header']); st.latex(r'''LCOH = \frac{CAPEX_{adj} + \sum_{t=1}^{n} \frac{OPEX_t + Repl_t}{(1+r)^t} + \frac{Decomm.}{(1+r)^n} + WC}{\sum_{t=1}^{n} H_{2,t}}'''); st.latex(r'''NPV = \sum_{t=0}^{n} \frac{FCF_t}{(1+r)^t}''')

st.header(T['analysis_header'])
tab_f1, tab_f2, tab_f3, tab_f4 = st.tabs([T['tab_cost_breakdown'], T['tab_opex_evolution'], T['tab_cumulative_cashflow'], T['environmental_header']])
with tab_f1:
    st.plotly_chart(plot_cost_breakdown_bar(cashflows_df, params, T), use_container_width=True)
with tab_f2:
    st.plotly_chart(plot_opex_evolution(cashflows_df, T), use_container_width=True)
with tab_f3:
    st.plotly_chart(plot_cumulative_cashflow(cashflows_df, kpi_results, T), use_container_width=True)
with tab_f4:
    st.plotly_chart(plot_co2_avoided(cashflows_df, T), use_container_width=True)

st.header(T['risk_header'])
tab_r1, tab_r2, tab_r3 = st.tabs([T['tab_tornado'], T['tab_heatmap'], T['tab_monte_carlo']])

with tab_r1:
    st.info("The Tornado plot shows which variables have the greatest impact on profitability (NPV). The longer the bar, the more sensitive the variable.")
    if st.button(T['tornado_button']):
        tornado_df = run_tornado_analysis(params, T)
        st.plotly_chart(plot_tornado(tornado_df, kpi_results['npv'], T), use_container_width=True)

with tab_r2:
    st.info("The heatmap allows you to see the combined effect of two variables on the NPV. Green means a positive NPV (profit) and red means a negative NPV (loss).")
    heatmap_vars = {
        T['h2_sale_price']: 'h2_sale_price',
        T['adjusted_capex']: 'adjusted_capex',
        T['elec_price']: 'elec_price_mwh',
        T['total_fixed_opex']: 'base_total_fixed_opex'
    }
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        var1_name = st.selectbox(T['heatmap_var1'], options=list(heatmap_vars.keys()), index=0)
    with col_h2:
        var2_name = st.selectbox(T['heatmap_var2'], options=list(heatmap_vars.keys()), index=2)
    
    if st.button(T['heatmap_button']):
        if var1_name == var2_name:
            st.warning("Please select two different variables.")
        else:
            var1_key = heatmap_vars[var1_name]
            var2_key = heatmap_vars[var2_name]
            npv_matrix, x_vals, y_vals = run_heatmap_analysis(params, var1_key, var2_key)
            st.plotly_chart(plot_heatmap(npv_matrix, x_vals, y_vals, T, var1_name, var2_name), use_container_width=True)

with tab_r3:
    st.info(T['mc_explanation'])
    if st.button(T['mc_button']):
        with st.spinner("Simulation in progress... (may take 30-60 seconds)"):
            mc_results = run_monte_carlo(params)
            prob_success = (mc_results > 0).mean() * 100
            col_mc1, col_mc2 = st.columns([2, 1])
            with col_mc1:
                fig_mc = go.Figure(data=[go.Box(x=mc_results, name='NPV Distribution', marker_color='#4C78A8', boxpoints='all', jitter=0.3, pointpos=-1.8)])
                fig_mc.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even point")
                fig_mc.update_layout(title_text=T['mc_title'], xaxis_title="Net Present Value (NPV) in €", template="plotly_dark", height=400)
                st.plotly_chart(fig_mc, use_container_width=True)
            with col_mc2:
                st.metric(T['mc_prob_success'], f"{prob_success:.2f} %")
                fig_pie = plot_monte_carlo_pie(prob_success, T)
                st.plotly_chart(fig_pie, use_container_width=True)

st.header(T['real_options_header'])
st.info(T['real_options_desc'])
col_opt1, col_opt2 = st.columns(2)
col_opt1.metric(T['option_value_metric'], f"€{option_value:,.0f}", help=T['option_help'])
col_opt2.metric(T['total_value_metric'], f"€{kpi_results['npv'] + option_value:,.0f}", help=T['total_value_help'])
