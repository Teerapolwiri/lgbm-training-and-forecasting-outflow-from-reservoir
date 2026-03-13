#%%
import matplotlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import optuna
import os
from optuna.samplers import TPESampler
import warnings
import matplotlib.dates as mdates
import geopandas as gpd
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from functools import reduce
warnings.filterwarnings('ignore')
matplotlib.use('Agg')

os.makedirs('Model', exist_ok=True)
os.makedirs('pic', exist_ok=True)

def create_integrated_objective(inflow_arr, delta_v_arr, delta_t, o_upper, o_lower, mu1=1.0, mu2=0.5, mu3=0.8):
    """
    inflow_arr : Array of inflow discharge (m^3/s)
    delta_v_arr: Array of actual water volume change (m^3)
    delta_t    : Duration of 1 time step (seconds)
    o_upper    : Maximum outflow limit (m^3)
    o_lower    : Minimum outflow limit (m^3)
    mu1, mu2, mu3 : Weight coefficients for each constraint
    """
    def integrated_loss(preds, train_data):
        labels = train_data.get_label()
        
        # L1: Prediction Accuracy (MSE)
        grad_l1 = preds - labels
        hess_l1 = np.ones_like(preds)
        
        # L2: Water Balance Constraint
        # Equation: ((Inflow - Outflow) * delta_t) - Delta_V       
        scale_factor = 1e-6
        
        phys_error = ((inflow_arr - preds) * delta_t * scale_factor) - (delta_v_arr * scale_factor)
        
        # Derivative with respect to preds (Outflow)
        grad_l2 = - (delta_t * scale_factor) * phys_error
        hess_l2 = np.ones_like(preds) * ((delta_t * scale_factor) ** 2)
               
        # L3: Boundary Constraints       
        grad_l3 = np.zeros_like(preds)
        hess_l3 = np.zeros_like(preds)
        
        # Upper Rule Curve
        preds_rule = preds * 11.57
        over_idx = preds_rule > o_upper
        grad_l3[over_idx] = preds_rule[over_idx] - o_upper
        hess_l3[over_idx] = 1.0
        
        # Lower Rule Curve
        under_idx = preds_rule < o_lower
        grad_l3[under_idx] = preds_rule[under_idx] - o_lower
        hess_l3[under_idx] = 1.0
        
        # Combine Gradient and Hessian with weighted mu coefficients       
        final_grad = (mu1 * grad_l1) + (mu2 * grad_l2) + (mu3 * grad_l3)
        final_hess = (mu1 * hess_l1) + (mu2 * hess_l2) + (mu3 * hess_l3)
        
        return final_grad, final_hess

    return integrated_loss

#BAYESIAN OPTIMIZATION WITH TPE
def make_objective(train_data, test_data, x_test, y_test, custom_obj):
    def objective_function(trial):
        """
        Objective function for Bayesian Optimization using TPE sampler.
        Optimizes LightGBM hyperparameters to minimize RMSE.
        """
        # Define hyperparameter search space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 150),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 20),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'feature_pre_filter': False,
            'verbose': -1,
            'seed': 42,
            'objective': 'custom'
        }
        
        # Train model with suggested parameters
        gbm = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        fobj=custom_obj,
        valid_sets=[train_data, test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=True),
            lgb.log_evaluation(period=0)
        ]
        )

        y_pred = gbm.predict(x_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return rmse
    return objective_function

def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df.reset_index(drop=True)
#%%
rsv_dict = {'100104': 'Mae Kuang Udom Thara Dam',
 'rsv21': 'Aemsrwy Reservoir',
 'rsv53': 'Aemtam Reservoir',
 'rsv54': 'Aempuuem Reservoir',
 '200103': 'Mae Ngat Somboon Chon Dam',
 '100105': 'Kio Lom Dam',
 '200102': 'Sirikit Dam',
 '200101': 'Bhumibol Dam',
 'rsv502': 'Namely Reservoir',
 '100201': 'Huai Luang Dam',
 '200203': 'Nam Phung Dam'}

id_list = list(rsv_dict.keys())
name = list(rsv_dict.values())
print("\nReservoir IDs:", id_list)
print("\nReservoir Names:", name)

df_all = []
for i, n in zip(id_list, name):
    df = None
    if i.startswith('rsv'):
        print(f"\nFetching data for middle reservoir: {n} (ID: {i})")
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"API URL for middle reservoir with ID" # Replace with actual API URL
        print(f"Requesting URL: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            rule_curve_list = data.get("rule_curve", [])
            rsv_data_list = data.get("q_daily", [])
            rsv_name = data.get("dam_name", [])
            capacity = data.get("cap_resv", [])
            dead = data.get("low_qdisc", [])

            #rule
            df_rule_curve = pd.DataFrame(
                [{"date": item["date"],
                f"{i}_upper mcm": item["upper"], 
                f"{i}_lower mcm": item["lower"]} 
                for item in rule_curve_list])
            
            for col in [f"{i}_upper mcm", f"{i}_lower mcm"]:
                df_rule_curve[col] = pd.to_numeric(df_rule_curve[col])
            df_rule_curve['date'] = pd.to_datetime(df_rule_curve['date'])
            start_date = df_rule_curve['date'].min()
            end_date = start_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)
            mask = (df_rule_curve['date'] >= start_date) & (df_rule_curve['date'] <= end_date)
            df_rule_curve = df_rule_curve[mask]

            #dam data
            df_dam_data = pd.DataFrame(
                [{"date": item["date"],
                f"{i}_volume mcm": item["qdisc"], 
                f"{i}_inflow mcm": item["q_info"],
                f"{i}_outflow mcm": item["q_outfo"]} 
                for item in rsv_data_list])

            cols_to_convert = df_dam_data.columns.drop('date')
            df_dam_data[cols_to_convert] = df_dam_data[cols_to_convert].apply(pd.to_numeric)
            
            df_dam_data['date'] = pd.to_datetime(df_dam_data['date'])
            #merge rule and dam
            df_dam_data['month_day'] = df_dam_data['date'].dt.strftime('%m-%d')
            df_rule_curve['month_day'] = df_rule_curve['date'].dt.strftime('%m-%d')
            df = pd.merge(df_dam_data, df_rule_curve, on='month_day', how='left')
            df = df.drop(columns=['date_y', 'month_day'])
            df = df.rename(columns={'date_x': 'date'})
            df = df.sort_values('date').reset_index(drop=True)

            valid_data = df[df[f'{i}_outflow mcm'] != 0]
            first_idx = valid_data.index[0]
            last_idx = valid_data.index[-1]
            df = df.loc[first_idx:last_idx].reset_index(drop=True)
        else:
            print(f"Status Code: {response.status_code}")
    else:
        print(f"\nFetching data for large reservoir: {n} (ID: {i})")
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"API URL for large reservoir with ID" # Replace with actual API URL
        response = requests.get(url)
        print(f"Requesting URL: {url}")
        if response.status_code == 200:
            data = response.json()
            
            rule_curve_list = data.get("rule_curve", [])
            dam_data_list = data.get("dam_data", [])
            dam_name = data.get("dam_name", [])
            capacity = data.get("q_store", [])
            active_capacity = data.get("q_useless", [])

            #rule
            df_rule_curve = pd.DataFrame(
                [{"date": item["date"],
                f"{i}_upper mcm": item["upper"], 
                f"{i}_lower mcm": item["lower"]} 
                for item in rule_curve_list])
            
            for col in [f"{i}_upper mcm", f"{i}_lower mcm"]:
                df_rule_curve[col] = pd.to_numeric(df_rule_curve[col])
            df_rule_curve['date'] = pd.to_datetime(df_rule_curve['date'])
            start_date = df_rule_curve['date'].min()
            end_date = start_date + pd.DateOffset(years=1) - pd.Timedelta(days=1)
            mask = (df_rule_curve['date'] >= start_date) & (df_rule_curve['date'] <= end_date)
            df_rule_curve = df_rule_curve[mask]

            #dam data
            df_dam_data = pd.DataFrame(
                [{"date": item["date"],
                f"{i}_volume mcm": item["q_use"], 
                f"{i}_inflow mcm": item["inflow"],
                f"{i}_outflow mcm": item["outflow"]} 
                for item in dam_data_list])
            
            cols_to_convert = df_dam_data.columns.drop('date')
            df_dam_data[cols_to_convert] = df_dam_data[cols_to_convert].apply(pd.to_numeric)
            
            #รวม rule กับ dam
            df_dam_data['date'] = pd.to_datetime(df_dam_data['date'])
            df_dam_data['month_day'] = df_dam_data['date'].dt.strftime('%m-%d')
            df_rule_curve['month_day'] = df_rule_curve['date'].dt.strftime('%m-%d')
            df = pd.merge(df_dam_data, df_rule_curve, on='month_day', how='left')
            df = df.drop(columns=['date_y', 'month_day'])
            df = df.rename(columns={'date_x': 'date'})
            df = df.sort_values('date').reset_index(drop=True)

            valid_data = df[df[f'{i}_outflow mcm'] != 0]
            first_idx = valid_data.index[0]
            last_idx = valid_data.index[-1]
            df = df.loc[first_idx:last_idx].reset_index(drop=True)
        else:
            print(f"Status Code: {response.status_code}")
        
    if df is not None:
        df_all.append(df)
    # df.to_csv(f"data\{n}_data.csv", index=False)

df_all = reduce(
    lambda left, right: pd.merge(left, right, on='date', how='outer'), 
    df_all
)
df_all = df_all.sort_values('date').reset_index(drop=True)
# %%
for dam, dam_id in zip(name, id_list):
    print(f"\nProcessing data for {dam} (ID: {dam_id})")
    required_cols = ["date", f'{dam_id}_inflow mcm', f'{dam_id}_volume mcm', 
                     f'{dam_id}_outflow mcm', f'{dam_id}_upper mcm', f'{dam_id}_lower mcm']
    
    missing = [col for col in required_cols if col not in df_all.columns]
    if missing:
        print(f"Warning: No columns: {missing}")
        continue
    
    df = df_all[required_cols].dropna().reset_index(drop=True)
    numerical_columns = [col for col in df.columns if col != 'date']

    df = remove_outliers_iqr(df, numerical_columns)

    delta_t = 24 * 60 * 60

    conv = 10**6 / delta_t

    df['inflow cms']  = df[f'{dam_id}_inflow mcm']  * conv
    df['outflow cms'] = df[f'{dam_id}_outflow mcm'] * conv
    df['volume mcm']  = df[f'{dam_id}_volume mcm']

    feature = ['inflow cms', 'volume mcm', 'outflow cms']
    x = df[feature].copy()
    y = df['outflow cms']

    dv = (df[f'{dam_id}_inflow mcm'] - df[f'{dam_id}_outflow mcm']) * 1e6

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    delta_v_train, delta_v_test = train_test_split(dv, test_size=0.2, shuffle=False)

    # Load daily rule curve
    upper_arr = df[f'{dam_id}_upper mcm'].values
    lower_arr = df[f'{dam_id}_lower mcm'].values

    upper_series = pd.Series(upper_arr, index=df.index)
    lower_series = pd.Series(lower_arr, index=df.index)

    upper_rule_arr_train = upper_series.loc[x_train.index].values
    lower_rule_arr_train = lower_series.loc[x_train.index].values
    upper_rule_arr_test = upper_series.loc[x_test.index].values
    lower_rule_arr_test = lower_series.loc[x_test.index].values

    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

    lower_rule = lower_rule_arr_test.min()
    upper_rule = upper_rule_arr_test.max()

    inflow_train_arr = x_train[f'{dam_id}_inflow mcm'].values
    delta_v_train_arr = dv.iloc[x_train.index].values

    custom_obj = create_integrated_objective(
        inflow_arr=inflow_train_arr,
        delta_v_arr=delta_v_train_arr,
        delta_t=delta_t,
        o_upper=upper_rule_arr_train,
        o_lower=lower_rule_arr_train,
        mu1=1.0, mu2=0.5, mu3=0.8
    )

    pmp = TPESampler(seed=42, multivariate=True)
    pmp = optuna.create_study(
        direction='minimize',
        sampler=pmp,
        study_name=f'{str(dam)}_LightGBM-HPO'
    )

    objective = make_objective(train_data, test_data, x_test, y_test, custom_obj)
    pmp.optimize(objective, n_trials=2000, show_progress_bar=True)
    
    best_params = pmp.best_params
    best_rmse = pmp.best_value

    print(f"Best RMSE achieved: {best_rmse:.6f}")
    print("\nBest Parameters:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"  {param:20s} : {value:.6f}")
        else:
            print(f"  {param:20s} : {value}")

    final_params = {
        'learning_rate': best_params['learning_rate'],
        'num_leaves': best_params['num_leaves'],
        'max_depth': best_params['max_depth'],
        'min_data_in_leaf': best_params['min_data_in_leaf'],
        'bagging_fraction': best_params['bagging_fraction'],
        'bagging_freq': best_params['bagging_freq'],
        'lambda_l1': best_params['lambda_l1'],
        'lambda_l2': best_params['lambda_l2'],
        'feature_fraction': best_params['feature_fraction'],
        'feature_pre_filter': False,
        'verbose': -1,
        'seed': 42,
        'objective': 'regression'
    }

    pmp_model = lgb.train(
        final_params,
        train_data,
        fobjective=custom_obj,
        num_boost_round=1000,
        valid_sets=[train_data, test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=20)
        ]
    )

    pmp_model.save_model(f'Model\lgb_model_{str(dam_id)}.txt')
    y_pred = pmp_model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nFinal RMSE Value (on Test Set): {rmse:.4f}")

    try:
        upper_rule_display_min = upper_rule_arr_test.min()
        upper_rule_display_max = upper_rule_arr_test.max()
        lower_rule_display_min = lower_rule_arr_test.min()
        lower_rule_display_max = lower_rule_arr_test.max()
    except NameError:
        upper_rule_display_min = upper_rule_display_max = upper_rule
        lower_rule_display_min = lower_rule_display_max = lower_rule

    #VISUALIZATION: OPTUNA OPTIMIZATION HISTORY
    fig_opt = plt.figure(figsize=(18, 12))

    # 1. Optimization History (RMSE over trials)
    ax_hist = plt.subplot(2, 3, 1)
    trials_data = pmp.trials
    trial_numbers = [t.number + 1 for t in trials_data]
    trials_data   = [t for t in pmp.trials if t.value is not None]
    trial_numbers = [t.number + 1 for t in trials_data]
    trial_values  = [t.value for t in trials_data]
    ax_hist.plot(trial_numbers, trial_values, 'b-o', linewidth=1, markersize=2)
    ax_hist.axhline(y=best_rmse, color='r', linestyle='--', linewidth=1, label=f'Best RMSE: {best_rmse:.4f}')
    ax_hist.set_xlabel('Trial Number', fontsize=11)
    ax_hist.set_ylabel('RMSE', fontsize=11)
    ax_hist.set_title(f'Bayesian Optimization History for {dam}', fontsize=12, fontweight='bold')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    # 2. Parameter Importance (Learning Rate)
    ax_lr = plt.subplot(2, 3, 2)
    lr_values = [t.params['learning_rate'] for t in trials_data]
    ax_lr.scatter(lr_values, trial_values, alpha=0.6, s=50, c=trial_numbers, cmap='viridis')
    ax_lr.axvline(x=best_params['learning_rate'], color='r', linestyle='--', linewidth=1)
    ax_lr.set_xlabel('Learning Rate', fontsize=11)
    ax_lr.set_ylabel('RMSE', fontsize=11)
    ax_lr.set_title(f'Learning Rate vs RMSE for {dam}', fontsize=12, fontweight='bold')
    ax_lr.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax_lr.collections[0], ax=ax_lr)
    cbar.set_label('Trial', fontsize=10)

    # 3. Parameter Importance (Num Leaves)
    ax_nl = plt.subplot(2, 3, 3)
    nl_values = [t.params['num_leaves'] for t in trials_data]
    ax_nl.scatter(nl_values, trial_values, alpha=0.6, s=100, c=trial_numbers, cmap='plasma')
    ax_nl.axvline(x=best_params['num_leaves'], color='r', linestyle='--', linewidth=1)
    ax_nl.set_xlabel('Number of Leaves', fontsize=11)
    ax_nl.set_ylabel('RMSE', fontsize=11)
    ax_nl.set_title(f'Num Leaves vs RMSE for {dam}', fontsize=12, fontweight='bold')
    ax_nl.grid(True, alpha=0.3)
    plm = plt.colorbar(ax_nl.collections[0], ax=ax_nl)
    plm.set_label('Trial', fontsize=10)

    # 4. Best Parameters Panel
    ax_bp = plt.subplot(2, 3, 5)
    ax_bp.axis('off')
    params_text = f"Best Hyperparameters Found for {dam}:\n\n"
    for i, (param, value) in enumerate(best_params.items()):
        if isinstance(value, float):
            params_text += f"{param:20s} : {value:.6f}\n"
        else:
            params_text += f"{param:20s} : {value}\n"

    ax_bp.text(0.05, 0.95, params_text, transform=ax_bp.transAxes,
                fontfamily='monospace', fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 5. Predictions vs Actual (with optimized model)
    ax_prv = plt.subplot(2, 3, 4)
    ax_prv.scatter(y_test, y_pred, alpha=0.5, s=30, label='compare 2 variable', color='red')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax_prv.plot(lims, lims, 'k--', lw=2, label='Perfect Prediction')
    ax_prv.set_xlabel('Actual Value (m³/s)', fontsize=11)
    ax_prv.set_ylabel('Predicted Value (m³/s)', fontsize=11)
    ax_prv.set_title(f'Optimized Model: Predictions vs Actual for {dam}', fontsize=12, fontweight='bold')
    ax_prv.legend(fontsize=9)
    ax_prv.grid(True, alpha=0.3)

    # 6. Performance Summary
    ax_ps = plt.subplot(2, 3, 6)
    ax_ps.axis('off')
    summary_text = f"""
    Optimization Summary for {dam} Dam

    Optimization Settings:
        - Algorithm: Bayesian Optimizer
        - Number of Trials: 2000
        - Best RMSE: {best_rmse:.6f}

    Final Model Performance:
        - Test RMSE: {rmse:.6f}
        
    Model Statistics:
        - Samples: {len(y_test)}
        - Min Error: {np.min(np.abs(y_test.values - y_pred)):.4f}
        - Max Error: {np.max(np.abs(y_test.values - y_pred)):.4f}
        - Mean Error: {np.mean(np.abs(y_test.values - y_pred)):.4f}
    """
    ax_ps.text(0.05, 0.95, summary_text, transform=ax_ps.transAxes,
                fontfamily='monospace', fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(rf'pic\bayesian_optimization_results_{str(dam)}.png', dpi=330, bbox_inches='tight')
    #plt.show()
    plt.close()

    #Plot
    fig = plt.figure(figsize=(18, 12))

    # 1. Predictions vs Actual (Before Clip)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_test, y_pred, alpha=0.5, s=30, label='Predicted', color='red')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, 'k--', lw=2, label='Perfect Prediction')
    ax1.plot([y_test.min(), y_test.max()], 
                [y_test.min()*1.3, y_test.max()*1.3], color='red', linestyle='--')
    ax1.plot([y_test.min(), y_test.max()], 
                [y_test.min()*0.7, y_test.max()*0.7], color='red', linestyle='--')
    ax1.set_xlabel('Actual Value (m³/s)', fontsize=10)
    ax1.set_ylabel('Predicted Value (m³/s)', fontsize=10)
    ax1.set_title(f'Predictions vs Actual Values for {dam}', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Residuals Error (Deviation from actual values)
    ax2 = plt.subplot(2, 3, 1)
    residuals = y_test.values - y_pred
    ax2.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero Error')
    ax2.set_xlabel('Residuals (Actual - Predicted)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title(f'Distribution of Residuals for {dam}', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Predictions Before vs After Clip
    ax3 = plt.subplot(2, 3, 3)
    x_indices = pd.to_datetime(df.loc[y_test.index, 'date']).values
    n_plot = len(y_test)
    inflow_mcm = df.loc[x_test.index, 'inflow mcm']
    dv_each_step = inflow_mcm - (y_pred * 0.0864) # convert from m^3/s to mcm/day
    dv_qout = inflow_mcm - (y_test * 0.0864) # convert from m^3/s to mcm/day
    com_dv = np.cumsum(dv_each_step)
    com_dv_qout = np.cumsum(dv_qout)
    initial_volume = df.loc[x_test.index[0], 'volume mcm']
    v_pred = initial_volume + com_dv
    v_qout = initial_volume + com_dv_qout
    ax3.plot(x_indices[:n_plot],df['upper'][len(df)-n_plot:], 'r-', alpha=0.7, linewidth=1.5, label='URC')
    ax3.plot(x_indices[:n_plot],df['lower'][len(df)-n_plot:], 'r-', alpha=0.7, linewidth=1.5, label='LRC')
    ax3.plot(x_indices[:n_plot], v_qout[:n_plot], 'b-', alpha=0.7, linewidth=1, label='Volume obs')
    ax3.plot(x_indices[:n_plot], v_pred[:n_plot], 'g-', alpha=0.7, linewidth=1, label='Volume prediction')
    ax3.set_xlabel(f'Sample Index (first {n_plot})', fontsize=10)
    ax3.set_ylabel('Volume (mcm)', fontsize=10)
    ax3.set_title(f'Effect of Rule Curve on Volume for {dam}', fontsize=11, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Absolute Errors
    ax4 = plt.subplot(2, 3, 2)
    abs_errors = np.abs(y_test.values - y_pred)
    ax4.scatter(y_test, abs_errors, alpha=0.6, s=30, color='darkred')
    ax4.set_xlabel('Actual Value (m³/s)', fontsize=10)
    ax4.set_ylabel('Absolute Error (m³/s)', fontsize=10)
    ax4.set_title(f'Absolute Errors vs Actual Values for {dam}', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Time Series Comparison
    ax5 = plt.subplot(2, 3, 3)
    # n_samples = min(365, len(y_test))
    n_samples = len(y_test)
    dates = pd.to_datetime(df.loc[y_test.index, 'date']).values
    ax5.plot(dates[:n_samples], y_test.values[:n_samples], 'b-', alpha=0.7, linewidth=2, label='Actual')
    ax5.plot(dates[:n_samples], y_pred[:n_samples], 'r-', alpha=0.7, linewidth=2, label='Predicted')
    ax5.set_xlabel('Date', fontsize=10)
    ax5.set_ylabel('Outflow (m³/s)', fontsize=10)
    ax5.set_title(f'Time Series Comparison for {dam}', fontsize=11, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # 6. Statistics Summary
    ax6 = plt.subplot(2, 3, 4)
    ax6.axis('off')
    stats_text = f"""
    Model Evaluation Summary for {dam} Dam

    General Information:
        • Number of Test Samples: {len(y_test)}
        • RMSE Value: {rmse:.4f} m³/s
        
    Actual Values:
        • Min: {y_test.min():.2f} m³/s
        • Max: {y_test.max():.2f} m³/s
        • Mean: {y_test.mean():.2f} m³/s
        • Std: {y_test.std():.2f} m³/s
        
    Predicted Values:
        • Min: {y_pred.min():.2f} m³/s
        • Max: {y_pred.max():.2f} m³/s
        • Mean: {y_pred.mean():.2f} m³/s
        • Std: {y_pred.std():.2f} m³/s

    Rule Curve Status:
        • Upper Rule (min/max): {upper_rule_display_min:.2f} / {upper_rule_display_max:.2f} m³
        • Lower Rule (min/max): {lower_rule_display_min:.2f} / {lower_rule_display_max:.2f} m³
        
    Errors:
        • MAE: {np.mean(abs_errors):.4f} m³/s
        • Max Error: {np.max(abs_errors):.4f} m³/s
    """
    ax6.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=12,
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(rf'pic\{str(dam)}_reservoir_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# %%

