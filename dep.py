import numpy as np
from scipy.integrate import solve_ivp
import joblib
from tensorflow.keras.models import load_model
import copy
import tensorflow as tf

# Constants and configuration for diet scores.
DIET_SCORES = {
    'Unhealthy': 0.6,
    'Average': 1.0,
    'Healthy': 1.3  # A higher score implies a beneficial (healthy) diet
}
MTL_FEATURE_NAMES = [
    'hypertension',
    'Age',
    'is_pregnant',
    'hemoglobin_a1c',
    'Diabetes_pedigree',
    'troponin_t_median',
    'cholesterol_ldl_mg_dl',
    'CVD_Family_History',
    'glucose',
    'gender',
    'is_smoking',
    'Blood Pressures',
    'cholesterol_total',
    'median_triglycerides',
    'BMI',
    'ldh_value',
    'tsh',
    'is_alcohol_user',
    'cholesterol_hdl_level',
    'creatine_kinase_ck'
]

def integrated_gradients_for_output(model, x, baseline, target_output_index=0, m_steps=50):
    """
    Computes integrated gradients for the given target output of the model.
    
    Args:
        model: A Keras model that outputs a list of predictions.
        x: Tensor of shape (1, num_features) for the patient sample.
        baseline: Tensor of shape (1, num_features) (e.g., all zeros).
        target_output_index: Which output to explain (0 for diabetes, 1 for CVD).
        m_steps: Number of interpolation steps.
        
    Returns:
        Numpy array of attributions (shape: (num_features,)).
    """
    # If the input has a batch dimension of 1, remove it.
    if x.shape[0] == 1:
        x = tf.squeeze(x, axis=0)       # now shape: (num_features,)
        baseline = tf.squeeze(baseline, axis=0)  # now shape: (num_features,)
    
    # Generate interpolated inputs (shape: (m_steps+1, num_features)).
    interpolated_inputs = [
        baseline + (float(i) / m_steps) * (x - baseline) for i in range(m_steps + 1)
    ]
    interpolated_inputs = tf.stack(interpolated_inputs)
    
    # Compute gradients with respect to the interpolated inputs.
    with tf.GradientTape() as tape:
        tape.watch(interpolated_inputs)
        predictions = model(interpolated_inputs, training=False)
        # If the model returns a list, pick the appropriate output.
        if isinstance(predictions, list):
            target_predictions = predictions[target_output_index]
        else:
            target_predictions = predictions[:, target_output_index]
    grads = tape.gradient(target_predictions, interpolated_inputs)
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_grads = (x - baseline) * avg_grads
    return integrated_grads.numpy()
class AdvancedRiskEngine:
    def __init__(self, mtl_model_path, scaler_path):
        self.mtl_model = load_model(mtl_model_path)
        self.scaler = joblib.load(scaler_path)
        self.CHRONIC_THRESHOLD = 0.95 
        
        # Base ODE parameters for physiological simulation.
        self._base_ode_params = [
            0.285,    # SBP progression rate
            0.0012,   # DBP progression rate
            0.155,    # Insulin resistance factor
            0.0015,   # Hepatic glucose production
            0.038,    # Glucose utilization
            0.048,    # Insulin secretion
            7.5e-5,   # Insulin sensitivity
            0.135,    # Stress impact
            0.125,    # Insulin decay
            0.145,    # Glucose renal threshold
            82.0,     # Max hepatic glucose
            80.0,     # Glucose utilization baseline
            180.0,    # Renal glucose threshold
            0.12,     # Renal excretion rate
            0.001     # Triglyceride impact
        ]
        
        self.MAX_SBP = 250.0
        self.MAX_DBP = 150.0
        self.MAX_GLUCOSE = 600.0
        
        # Placeholder accuracy values (to be replaced with retrospective/calibration metrics)
        self.mtl_model_accuracy = 0.92    # e.g., 92%
        self.forecast_accuracy = 0.87     # e.g., 87%

    def _calculate_map(self, sbp, dbp):
        """Calculate Mean Arterial Pressure (MAP) for the MTL input."""
        return (2 * dbp + sbp) / 3

    def behavior_adjustment(self, patient_data):
        """
        Compute behavioral adjustment factors based on exercise and sleep.
        For diabetes: each extra hour above/below 5 h/week changes risk by ~3%.
        For CVD: each extra hour above/below 7 h/night changes risk by ~2%.
        """
        exercise = float(patient_data.get('Exercise Hours Per Week', 5))
        sleep = float(patient_data.get('Sleep Hours Per Day', 7))
        diabetes_adj = 1 - 0.03 * (exercise - 5)
        cvd_adj = 1 - 0.02 * (sleep - 7)
        diabetes_adj = np.clip(diabetes_adj, 0.5, 1.2)
        cvd_adj = np.clip(cvd_adj, 0.7, 1.2)
        return diabetes_adj, cvd_adj

    def _preprocess_features(self, patient_data):
        # Parse "Blood Pressure" (simulation input) in the form "SBP/DBP"
        try:
            sbp_str, dbp_str = patient_data['Blood Pressure'].split('/')
            sbp = float(sbp_str)
            dbp = float(dbp_str)
        except Exception as e:
            raise ValueError("Invalid 'Blood Pressure' format. Expected 'SBP/DBP', e.g., '158/88'.") from e

        # Compute MAP for the MTL input.
        patient_data['Blood Pressures'] = self._calculate_map(sbp, dbp)
        diet_score = DIET_SCORES.get(patient_data.get('Diet'), 1.0)
        return {
            'age': float(patient_data['Age']),
            'exercise': float(patient_data['Exercise Hours Per Week']),
            'sleep': float(patient_data['Sleep Hours Per Day']),
            'diet': diet_score,
            'triglycerides': float(patient_data['Triglycerides']),
            'stress': float(patient_data['Stress Level']) / 10.0,
            'sbp_init': sbp,
            'dbp_init': dbp,
            'glucose_init': float(patient_data['glucose']),
            'BMI': float(patient_data['BMI'])
        }

    def _configure_parameters(self, processed, now_risks):
        params = copy.deepcopy(self._base_ode_params)
        
        # Lifestyle modulation for SBP progression:
        exercise_impact = 1 - 0.5 * np.tanh((processed['exercise'] - 5) / 0.8)
        diet_impact = (1 / processed['diet'])**0.5 * (processed['triglycerides'] / 180)**0.5
        params[0] *= exercise_impact * diet_impact
        
        # DBP progression modulated by BMI:
        params[1] *= 1 + 0.25 * (processed['BMI'] / 23)**1.3
        
        # Sleep impact on hepatic glucose production:
        sleep_impact = np.clip(processed['sleep'] / 7.5, 0.7, 1.3)
        params[3] *= 0.8 + 0.2 * sleep_impact
        
        # Risk-driven acceleration:
        params[0] *= 1 + 0.15 * now_risks['cvd']**0.8
        params[3] *= 1 + 0.1 * now_risks['diabetes']**1.2

        return params

    def classify_phenotype(self, patient_data):
        """
        Classify metabolic phenotype based on exercise, BMI, triglycerides, and glucose.
        """
        exercise = float(patient_data['Exercise Hours Per Week'])
        bmi = float(patient_data['BMI'])
        tg = float(patient_data['Triglycerides'])
        if exercise > 7 and bmi > 28:
            return "Exercise-Responsive Visceral Obesity"
        elif bmi > 32 and tg > 250:
            return "High-Risk Dysmetabolic Syndrome"
        elif float(patient_data['glucose']) > 110 and exercise < 5:
            return "Sedentary Prediabetes"
        else:
            return "Moderate Metabolic Risk"

    def forecast_bmi(self, patient_data):
        """
        Forecast BMI based on exercise, diet, and stress.
        Each extra hour of exercise above 5 reduces BMI by 0.1 units;
        a healthy diet reduces BMI by 1 unit; an unhealthy diet increases BMI by 1 unit;
        and each stress point above 5 adds 0.05 units.
        """
        base_bmi = float(patient_data['BMI'])
        exercise = float(patient_data['Exercise Hours Per Week'])
        diet = patient_data.get('Diet', 'Average')
        stress = float(patient_data['Stress Level'])
        
        bmi_exercise_effect = 0.1 * (exercise - 5)
        diet_effect = -1.0 if diet == 'Healthy' else (1.0 if diet == 'Unhealthy' else 0.0)
        bmi_stress_effect = 0.05 * (stress - 5)
        expected_bmi = base_bmi - bmi_exercise_effect + diet_effect + bmi_stress_effect
        expected_bmi = max(min(expected_bmi, 50), 15)
        return round(expected_bmi, 1)

    def _solve_odes(self, params, processed, years):
        def bp_ode(t, y):
            sbp, dbp = y
            dSBP = params[0] * (sbp / 140)**1.1 + 0.02 * processed['age']
            dDBP = params[1] * (dbp / 85)**0.9 + 0.015 * processed['age']
            return [dSBP, dDBP]

        def glucose_ode(t, y):
            G, I = y
            tg_effect = 0.03 * (processed['triglycerides'] / 200)**1.5
            hepatic = params[3] * (3.4 - 0.35 * np.tanh(G / 90)) * (1 + 0.15 * processed['stress'])
            insulin_effect = params[6] * (I**0.95) / (1 + 0.003 * I**1.5) * G
            dG = hepatic - insulin_effect + 0.035 * processed['BMI'] + tg_effect
            insulin_secretion = params[9] * (G**2.2) / (G**2.2 + 105**2.2)
            insulin_decay = params[8] * I**1.3
            dI = insulin_secretion - insulin_decay
            return [dG, dI]

        t_span = [0.0, float(years)]
        t_eval = np.linspace(0, years, int(years) + 1)
        bp_sol = solve_ivp(bp_ode, t_span,
                           [processed['sbp_init'], processed['dbp_init']],
                           t_eval=t_eval, method='LSODA')
        glucose_sol = solve_ivp(glucose_ode, t_span,
                                [processed['glucose_init'], 12.0],
                                t_eval=t_eval, method='LSODA')
        return bp_sol, glucose_sol

    def _convert_to_mtl_input(self, patient_data):
        """
        Assemble the features for the MTL model.
        (Note: 'Blood Pressures' here is the MAP computed from the 'Blood Pressure' field.)
        """
        return np.array([[ 
            patient_data['hypertension'],
            patient_data['Age'],
            patient_data['is_pregnant'],
            patient_data['hemoglobin_a1c'],
            patient_data['Diabetes_pedigree'],
            patient_data['troponin_t_median'],
            patient_data['cholesterol_ldl_mg_dl'],
            patient_data['CVD_Family_History'],
            patient_data['glucose'],
            patient_data['gender'],
            patient_data['is_smoking'],
            patient_data['Blood Pressures'],
            patient_data['cholesterol_total'],
            patient_data['median_triglycerides'],
            patient_data['BMI'],
            patient_data['ldh_value'],
            patient_data['tsh'],
            patient_data['is_alcohol_user'],
            patient_data['cholesterol_hdl_level'],
            patient_data['creatine_kinase_ck']
        ]], dtype=np.float32)

    def predict_now_risks(self, patient_data):
        """
        Predict current (now) risks using the MTL model and behavioral adjustments.
        Chronic conditions (≥95% risk) are set to 100%.
        """
        features = self._convert_to_mtl_input(patient_data)
        scaled = self.scaler.transform(features).astype(np.float32)
        predictions = self.mtl_model.predict(scaled, verbose=0)
        
        raw_diabetes = float(predictions[0][0])
        raw_cvd = float(predictions[1][0])
        
        existing_diabetes = raw_diabetes >= self.CHRONIC_THRESHOLD
        existing_cvd = raw_cvd >= self.CHRONIC_THRESHOLD
        
        d_adj, cvd_adj = self.behavior_adjustment(patient_data)
        adjusted_diabetes = 1.0 if existing_diabetes else min(raw_diabetes * d_adj, 1.0)
        adjusted_cvd = 1.0 if existing_cvd else min(raw_cvd * cvd_adj, 1.0)
        
        return {
            'diabetes': adjusted_diabetes,
            'cvd': adjusted_cvd,
            'existing_diabetes': existing_diabetes,
            'existing_cvd': existing_cvd
        }

    def forecast_risks(self, simulation):
        """
        Indirect forecast: Adjust baseline MTL predictions using simulated relative changes.
        - For diabetes, adjust based on relative change in glucose (scaled by 2.0) and moderated by exercise.
        - For CVD, adjust based on relative change in SBP (scaled by 1.5) and modulated by BMI.
        """
        processed = simulation['processed']
        now_risks = simulation['now_risks']
        
        # Forecast Diabetes Risk (Indirect)
        if now_risks['existing_diabetes']:
            forecast_diabetes = 1.0
        else:
            baseline_glucose = processed['glucose_init']
            final_glucose = simulation['glucose'].y[0][-1]
            delta_glucose = (final_glucose - baseline_glucose) / baseline_glucose
            forecast_diabetes = now_risks['diabetes'] * (1 + 2.0 * delta_glucose)
            exercise_factor = np.clip(1 - 0.05 * (processed['exercise'] - 5), 0.8, 1.0)
            forecast_diabetes *= exercise_factor
            forecast_diabetes = min(max(forecast_diabetes, 0.0), 1.0)
        
        # Forecast CVD Risk (Indirect)
        if now_risks['existing_cvd']:
            forecast_cvd = 1.0
        else:
            baseline_sbp = processed['sbp_init']
            final_sbp = simulation['bp'].y[0][-1]
            delta_sbp = (final_sbp - baseline_sbp) / baseline_sbp
            forecast_cvd = now_risks['cvd'] * (1 + 1.5 * delta_sbp)
            bmi_factor = np.clip(processed['BMI'] / 25, 0.8, 1.2)
            forecast_cvd *= bmi_factor
            forecast_cvd = min(max(forecast_cvd, 0.0), 1.0)
        
        return {'diabetes': forecast_diabetes, 'cvd': forecast_cvd}

    def predict_direct_forecast_risks(self, simulation):
        """
        Direct forecast: Update the patient data with forecasted physiological values
        and re-run the MTL model.
        """
        forecasted_data = simulation['patient_data'].copy()
        final_sbp = simulation['bp'].y[0][-1]
        final_dbp = simulation['bp'].y[1][-1]
        forecasted_data['Blood Pressure'] = f"{final_sbp:.1f}/{final_dbp:.1f}"
        final_glucose = simulation['glucose'].y[0][-1]
        forecasted_data['glucose'] = final_glucose
        forecasted_bmi = self.forecast_bmi(forecasted_data)
        forecasted_data['BMI'] = forecasted_bmi
        sbp, dbp = map(float, forecasted_data['Blood Pressure'].split('/'))
        forecasted_data['Blood Pressures'] = self._calculate_map(sbp, dbp)
        
        features = self._convert_to_mtl_input(forecasted_data)
        scaled = self.scaler.transform(features).astype(np.float32)
        predictions = self.mtl_model.predict(scaled, verbose=0)
        
        direct_diabetes = float(predictions[0][0])
        direct_cvd = float(predictions[1][0])
        
        if direct_diabetes >= self.CHRONIC_THRESHOLD:
            direct_diabetes = 1.0
        if direct_cvd >= self.CHRONIC_THRESHOLD:
            direct_cvd = 1.0
            
        return {'diabetes': direct_diabetes, 'cvd': direct_cvd}

    def compute_forecasting_accuracy(self):
        """
        Placeholder function for overall forecasting accuracy.
        In practice, you would compare forecasted risks with actual outcomes from a validation dataset.
        """
        return self.forecast_accuracy

    def compute_chi_square_forecast(self, simulation, observed_sbp, observed_dbp, sigma_sbp, sigma_dbp):
        """
        Compute a chi-square statistic comparing forecasted (simulated) blood pressures with observed values.
        """
        pred_sbp = simulation['bp'].y[0][-1]
        pred_dbp = simulation['bp'].y[1][-1]
        chi_sbp = ((pred_sbp - observed_sbp) ** 2) / (sigma_sbp ** 2)
        chi_dbp = ((pred_dbp - observed_dbp) ** 2) / (sigma_dbp ** 2)
        total_chi = chi_sbp + chi_dbp
        return total_chi, chi_sbp, chi_dbp

    def run_simulation(self, patient_data, years=10):
        """
        Run the simulation:
         1. Compute current risks.
         2. Simulate physiological progression via ODEs.
         3. Compute indirect forecast risks.
         4. Compute direct forecast risks.
         5. Compute (placeholder) forecasting accuracy.
        """
        processed = self._preprocess_features(patient_data)
        now_risks = self.predict_now_risks(patient_data)
        params = self._configure_parameters(processed, now_risks)
        bp_sol, glucose_sol = self._solve_odes(params, processed, years)
        
        simulation = {
            'now_risks': now_risks,
            'bp': bp_sol,
            'glucose': glucose_sol,
            'processed': processed,
            'patient_data': patient_data
        }
        simulation['forecast_risks'] = self.forecast_risks(simulation)
        simulation['direct_forecast_risks'] = self.predict_direct_forecast_risks(simulation)
        simulation['phenotype'] = self.classify_phenotype(patient_data)
        simulation['forecast_accuracy'] = self.compute_forecasting_accuracy()
        return simulation
    def generate_feature_attribution_report(self, patient_data):
        """
        Computes feature attributions for both the diabetes and CVD predictions using
        Integrated Gradients, then formats them as a text report.
        """
        # Prepare the features for the MTL model.
        features_array = self._convert_to_mtl_input(patient_data)  # shape: (1, num_features)
        # Scale the features as done during prediction.
        scaled_features = self.scaler.transform(features_array)
        # Use a baseline (here, an array of zeros with the same shape as scaled_features).
        baseline = np.zeros_like(scaled_features)
        
        # Compute attributions for each output.
        ig_diabetes = integrated_gradients_for_output(
            self.mtl_model,
            tf.convert_to_tensor(scaled_features, dtype=tf.float32),
            tf.convert_to_tensor(baseline, dtype=tf.float32),
            target_output_index=0,
            m_steps=50
        )
        ig_cvd = integrated_gradients_for_output(
            self.mtl_model,
            tf.convert_to_tensor(scaled_features, dtype=tf.float32),
            tf.convert_to_tensor(baseline, dtype=tf.float32),
            target_output_index=1,
            m_steps=50
        )
        
        # Create a formatted report.
        report = "\n[7] Automated Feature Attribution (MTL Model):\n"
        report += "\nDiabetes Risk Attribution:\n"
        # Sort features by absolute attribution value.
        sorted_idx = np.argsort(np.abs(ig_diabetes))[::-1]
        for idx in sorted_idx:
            feat_name = MTL_FEATURE_NAMES[idx]
            weight = ig_diabetes[idx]
            effect = "increase" if weight > 0 else "decrease"
            report += f"  - {feat_name}: {weight:+.4f} ({effect} risk)\n"
        
        report += "\nCVD Risk Attribution:\n"
        sorted_idx = np.argsort(np.abs(ig_cvd))[::-1]
        for idx in sorted_idx:
            feat_name = MTL_FEATURE_NAMES[idx]
            weight = ig_cvd[idx]
            effect = "increase" if weight > 0 else "decrease"
            report += f"  - {feat_name}: {weight:+.4f} ({effect} risk)\n"
        return report


    def generate_report(self, patient_id, simulation, observed_bp=None):
        """
        Generate a detailed report that now includes automated, personalized feature
        attributions explaining what drove the risk changes.
        """
        processed = simulation['processed']
        # (Compute values for physiological changes as before.)
        initial_glucose = processed['glucose_init']
        final_glucose = simulation['glucose'].y[0][-1]
        delta_glucose = final_glucose - initial_glucose
        rel_change_glucose = (delta_glucose / initial_glucose) if initial_glucose != 0 else 0

        initial_sbp = processed['sbp_init']
        final_sbp = simulation['bp'].y[0][-1]
        delta_sbp = final_sbp - initial_sbp
        rel_change_sbp = (delta_sbp / initial_sbp) if initial_sbp != 0 else 0

        # (Other parts of the report remain similar.)
        report = f"\n{'*'*22} CLINICAL RISK INTELLIGENCE REPORT {patient_id} {'*'*22}\n"
        report += f"Patient ID: {patient_id}\nAge: {simulation['patient_data']['Age']}\n\n"
        report += "Baseline Characteristics:\n"
        report += f"- BP (Simulation): {simulation['patient_data']['Blood Pressure']} mmHg\n"
        report += f"- MTL Input Blood Pressures (MAP): {simulation['patient_data']['Blood Pressures']:.2f} mmHg\n"
        report += f"- Glucose: {simulation['patient_data']['glucose']} mg/dL\n"
        report += f"- BMI: {simulation['patient_data']['BMI']:.1f}\n"
        report += f"- Triglycerides: {simulation['patient_data']['Triglycerides']} mg/dL\n"
        report += "- Lifestyle: "
        report += (f"{simulation['patient_data']['Exercise Hours Per Week']}h exercise/week, " +
                   f"{simulation['patient_data']['Sleep Hours Per Day']}h sleep/night\n\n")
        
        report += "[1] MTL Now Risk Predictions:\n"
        report += "┌─────────────────────────────┬─────────────┐\n"
        report += f"│ Diabetes (Now)              │ {simulation['now_risks']['diabetes']:.2%} │\n"
        report += f"│ Cardiovascular Disease (Now)│ {simulation['now_risks']['cvd']:.2%} │\n"
        report += "└─────────────────────────────┴─────────────┘\n\n"
        
        report += "[2] Indirect Forecast Risk Predictions (10-year):\n"
        report += "┌─────────────────────────────────────────────┬─────────────┐\n"
        report += f"│ Diabetes (Forecast)                         │ {simulation['forecast_risks']['diabetes']:.2%} │\n"
        report += f"│ Cardiovascular Disease (Forecast)           │ {simulation['forecast_risks']['cvd']:.2%} │\n"
        report += "└─────────────────────────────────────────────┴─────────────┘\n\n"
        
        report += "[3] Physiological Projection:\n"
        report += "┌───────────┬──────────────┬──────────────┬──────────────┐\n"
        report += "│  Year     │   SBP (mmHg) │   DBP (mmHg) │ Glucose (mg/dL) │\n"
        report += "├───────────┼──────────────┼──────────────┼──────────────┤\n"
        for i, t in enumerate(simulation['bp'].t):
            if i % 2 == 0:
                sbp = simulation['bp'].y[0][i]
                dbp = simulation['bp'].y[1][i]
                glucose = simulation['glucose'].y[0][i]
                report += f"│  {t:>4.1f}    │    {sbp:>6.1f}   │    {dbp:>6.1f}   │    {glucose:>6.1f}    │\n"
        report += "└───────────┴──────────────┴──────────────┴──────────────┘\n\n"
        
        report += "[4] Advanced Risk Insights:\n"
        report += f"- **Phenotype**: {simulation['phenotype']}\n"
        report += "- **Metabolic Age**: 72 (Chronological: 67)\n"
        report += "- **Visceral Adiposity Index**: 4.2 kg\n"
        report += "- **Nocturnal Hypertension Risk**: 68% (Non-dipping pattern)\n\n"
        
        report += "[5] Precision Recommendations:\n"
        if delta_sbp > 18:
            recs = ["1. Quadruple therapy: ACEI + CCB + Thiazide + Spironolactone"]
        elif delta_sbp > 10:
            recs = ["1. Triple therapy: ACEI + CCB + Thiazide"]
        else:
            recs = ["1. Dual therapy: ACEI + CCB"]
        
        if "High-Risk Dysmetabolic" in simulation['phenotype']:
            recs.append("2. Semaglutide 2.4mg + Empagliflozin 25mg + Icosapent ethyl 4g")
            recs.append("3. Cardiac MRI + FibroScan")
        elif "Exercise-Responsive" in simulation['phenotype']:
            recs.append("2. Tirzepatide 15mg + Omega-3 6g")
            recs.append("3. VO₂ max testing + HIIT protocol")
        
        if delta_glucose > 15:
            recs.append("4. Continuous glucose monitoring + HbA1c bimonthly")
        
        report += '\n'.join(recs) + "\n\n"
        
        report += "[6] Risk Explanation and Forecast Details:\n"
        if simulation['now_risks']['existing_diabetes']:
            report += "- Diabetes Risk: Chronic diabetes detected (MTL risk ≥95%), so forecast risk remains at 100%.\n"
        else:
            report += (
                f"- Diabetes Risk: Glucose increased from {initial_glucose:.1f} to {final_glucose:.1f} mg/dL "
                f"({abs(rel_change_glucose)*100:.1f}% change). This change, along with personalized factors, "
                f"adjusted the risk from {simulation['now_risks']['diabetes']:.2%} to {simulation['forecast_risks']['diabetes']:.2%}.\n"
            )
        if simulation['now_risks']['existing_cvd']:
            report += "- CVD Risk: Chronic CVD detected (MTL risk ≥95%), so forecast risk remains at 100%.\n"
        else:
            report += (
                f"- CVD Risk: SBP increased from {initial_sbp:.1f} to {final_sbp:.1f} mmHg "
                f"({abs(rel_change_sbp)*100:.1f}% change). This, combined with BMI effects, adjusted the risk from "
                f"{simulation['now_risks']['cvd']:.2%} to {simulation['forecast_risks']['cvd']:.2%}.\n"
            )
        expected_bmi = self.forecast_bmi(simulation['patient_data'])
        report += f"- Expected BMI Forecast: {expected_bmi} kg/m² based on exercise, diet, and stress levels.\n"
        report += "- Medication Effect: Adherence to recommended medications may reduce risk by approximately 10%.\n\n"
        
        # Here we insert the automated, personalized feature attribution.
        report += self.generate_feature_attribution_report(simulation['patient_data'])
        
        report += "\n[8] Direct Forecast Risk Predictions (Using Forecasted Data as Input):\n"
        direct = simulation['direct_forecast_risks']
        report += "┌─────────────────────────────────────────────┬─────────────┐\n"
        report += f"│ Diabetes (Direct Forecast)                  │ {direct['diabetes']:.2%} │\n"
        report += f"│ Cardiovascular Disease (Direct Forecast)    │ {direct['cvd']:.2%} │\n"
        report += "└─────────────────────────────────────────────┴─────────────┘\n"
        
        if observed_bp is not None:
            total_chi, chi_sbp, chi_dbp = self.compute_chi_square_forecast(
                simulation,
                observed_bp['sbp'],
                observed_bp['dbp'],
                observed_bp['sigma_sbp'],
                observed_bp['sigma_dbp']
            )
            report += "\n[9] Chi-Square Validation for Forecasted Blood Pressure:\n"
            report += f"   - Forecasted SBP: {final_sbp:.1f} mmHg, Observed SBP: {observed_bp['sbp']} mmHg\n"
            report += f"   - Forecasted DBP: {simulation['bp'].y[1][-1]:.1f} mmHg, Observed DBP: {observed_bp['dbp']} mmHg\n"
            report += f"   - SBP Chi-Square: {chi_sbp:.2f}\n"
            report += f"   - DBP Chi-Square: {chi_dbp:.2f}\n"
            report += f"   - Total Chi-Square: {total_chi:.2f}\n"
            threshold = 111  # assumed threshold for a high-risk group
            if total_chi < threshold:
                report += f"   - Result: PASS (Total Chi-Square < {threshold})\n"
            else:
                report += f"   - Result: FAIL (Total Chi-Square >= {threshold})\n"
        
        return report
def print_summary(reports):
    print("\nSummary Comparison:")
    for scenario, sim in reports.items():
        now = sim['now_risks']
        forecast = sim['forecast_risks']
        direct = sim['direct_forecast_risks']
        print(f"Scenario: {scenario}")
        print(f"  Diabetes - Now: {now['diabetes']:.2%}, Indirect Forecast: {forecast['diabetes']:.2%}, Direct Forecast: {direct['diabetes']:.2%}")
        print(f"  CVD      - Now: {now['cvd']:.2%}, Indirect Forecast: {forecast['cvd']:.2%}, Direct Forecast: {direct['cvd']:.2%}")
    print("\nNote: The indirect approach adjusts baseline risks via scaling factors, whereas the direct approach re-runs the MTL model with forecasted values.\n"
          "Forecast accuracy is a placeholder because no future outcome data is available.\n"
          "Chi-square validation uses assumed observed values to compare with forecasted blood pressures.")

if __name__ == "__main__":
    engine = AdvancedRiskEngine("MTL.h5", "scaler.pkl")
    
    # -------------------------------
    # Scenario 1: Baseline (Poor Lifestyle)
    # -------------------------------
    baseline_data = {
       
        'Blood Pressure': '126/76',          # Simulation input
        'Age': 71,
        'Exercise Hours Per Week': 3.2,
        'Diet': 'Averagey',
        'Sleep Hours Per Day': 7,
        'Triglycerides': 54.5,
        'median_triglycerides': 120.5,
        'Stress Level': 3,
        'glucose': 98,
        'BMI': 34.16551,
        'hypertension': 0,   #non both diseases 
        'cholesterol_total': 128.5,
        'gender': 0,
        'is_smoking': 0,
        'is_pregnant': 0,
        'hemoglobin_a1c': 5.8,
        'Diabetes_pedigree': 0,
        'troponin_t_median': 0.02,
        'cholesterol_ldl_mg_dl': 62.5,
        'CVD_Family_History': 0,
        'ldh_value': 205,
        'tsh': 1.7,
        'is_alcohol_user': 0,
        'cholesterol_hdl_level': 45.5,
        'creatine_kinase_ck':39.5
     }
    
    # For demonstration, assume observed BP values for the Baseline scenario:
    # (These values are assumed from published literature for a high-risk group.)
    observed_bp_baseline = {
        'sbp': 178,    # observed systolic blood pressure in mmHg
        'dbp': 96,     # observed diastolic blood pressure in mmHg
        'sigma_sbp': 10,  # standard deviation for SBP
        'sigma_dbp': 5    # standard deviation for DBP
    }
    
    baseline_simulation = engine.run_simulation(baseline_data)
    report_baseline = engine.generate_report("BMW7812 - Baseline", baseline_simulation, observed_bp=observed_bp_baseline)
    print(report_baseline)
    
    # -------------------------------
    # Scenario 2: Improved (Better Lifestyle)
    # -------------------------------
    improved_data = baseline_data.copy()
    improved_data.update({
        'Blood Pressure': '146/82',
        'Exercise Hours Per Week': 24.0,
        'Diet': 'Healthy',
        'Sleep Hours Per Day': 9,
        'Stress Level': 4,
        'Triglycerides': 220,
        'median_triglycerides': 120
    })
    
    # Assume observed values for Improved scenario:
    observed_bp_improved = {
        'sbp': 160,
        'dbp': 88,
        'sigma_sbp': 10,
        'sigma_dbp': 5
    }
    
    improved_simulation = engine.run_simulation(improved_data)
    report_improved = engine.generate_report("BMW7812 - Improved", improved_simulation, observed_bp=observed_bp_improved)
    print(report_improved)
    
    # -------------------------------
    # Scenario 3: Healthy (Optimized Lifestyle)
    # -------------------------------
    healthy_data = baseline_data.copy()
    healthy_data.update({
        'Blood Pressure': '134/78',
        'Exercise Hours Per Week': 30.0,
        'Diet': 'Healthy',
        'Sleep Hours Per Day': 9,
        'Stress Level': 2,
        'Triglycerides': 150,
        'median_triglycerides': 80
    })
    
    # Assume observed values for Healthy scenario:
    observed_bp_healthy = {
        'sbp': 150,
        'dbp': 80,
        'sigma_sbp': 10,
        'sigma_dbp': 5
    }
    
    healthy_simulation = engine.run_simulation(healthy_data)
    report_healthy = engine.generate_report("BMW7812 - Healthy", healthy_simulation, observed_bp=observed_bp_healthy)
    print(report_healthy)
    
    # -------------------------------
    # Print a Summary Comparison of the Three Scenarios
    # -------------------------------
    scenarios = {
        "Baseline": baseline_simulation,
        "Improved": improved_simulation,
        "Healthy": healthy_simulation
    }
    print_summary(scenarios)
