import pandas as pd
import numpy as np
import altair as alt


def calculate_quadrant_stats(
    plot_df: pd.DataFrame,
    x_var: str,
    y_var: str,
    x_threshold: float,
    y_threshold: float
) -> pd.DataFrame:
    """
    Calculate the count of each class (red/green) per quadrant.
    
    Parameters:
    -----------
    plot_df : pd.DataFrame
        DataFrame with x_var, y_var, and 'color' columns
    x_var : str
        Name of the x-axis variable
    y_var : str
        Name of the y-axis variable
    x_threshold : float
        Threshold value for x-axis
    y_threshold : float
        Threshold value for y-axis
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with quadrant statistics
    """
    # Define quadrants
    # Quadrant 1 (upper-right): x > x_threshold AND y > y_threshold
    # Quadrant 2 (upper-left): x <= x_threshold AND y > y_threshold
    # Quadrant 3 (lower-left): x <= x_threshold AND y <= y_threshold
    # Quadrant 4 (lower-right): x > x_threshold AND y <= y_threshold
    
    def assign_quadrant(row):
        if row[x_var] > x_threshold and row[y_var] > y_threshold:
            return 'Q1: Superior-Derecha'
        elif row[x_var] <= x_threshold and row[y_var] > y_threshold:
            return 'Q2: Superior-Izquierda'
        elif row[x_var] <= x_threshold and row[y_var] <= y_threshold:
            return 'Q3: Inferior-Izquierda'
        else:  # x > x_threshold and y <= y_threshold
            return 'Q4: Inferior-Derecha'
    
    plot_df = plot_df.copy()
    plot_df['quadrant'] = plot_df.apply(assign_quadrant, axis=1)
    
    # Count by quadrant and color
    stats = plot_df.groupby(['quadrant', 'color']).size().reset_index(name='count')
    
    # Pivot to have red and green as columns
    stats_pivot = stats.pivot(index='quadrant', columns='color', values='count').fillna(0)
    
    # Ensure both red and green columns exist
    if 'red' not in stats_pivot.columns:
        stats_pivot['red'] = 0
    if 'green' not in stats_pivot.columns:
        stats_pivot['green'] = 0
    
    # Add total column
    stats_pivot['Total'] = stats_pivot['red'] + stats_pivot['green']
    
    # Calculate percentages for each quadrant
    stats_pivot['% Rojo'] = (stats_pivot['red'] / stats_pivot['Total'] * 100).round(1)
    stats_pivot['% Verde'] = (stats_pivot['green'] / stats_pivot['Total'] * 100).round(1)
    
    # Handle division by zero (when Total is 0)
    stats_pivot['% Rojo'] = stats_pivot['% Rojo'].fillna(0)
    stats_pivot['% Verde'] = stats_pivot['% Verde'].fillna(0)
    
    # Reorder columns
    stats_pivot = stats_pivot[['red', 'green', 'Total', '% Rojo', '% Verde']]
    
    # Reset index to make quadrant a column
    stats_pivot = stats_pivot.reset_index()
    
    # Rename columns for display
    stats_pivot.columns = ['Cuadrante', 'Rojo (≤0)', 'Verde (>0)', 'Total', '% Rojo', '% Verde']
    
    # Ensure quadrants are in order
    quadrant_order = ['Q1: Superior-Derecha', 'Q2: Superior-Izquierda', 'Q3: Inferior-Izquierda', 'Q4: Inferior-Derecha']
    stats_pivot['Cuadrante'] = pd.Categorical(stats_pivot['Cuadrante'], categories=quadrant_order, ordered=True)
    stats_pivot = stats_pivot.sort_values('Cuadrante').reset_index(drop=True)
    
    return stats_pivot


def plot_text_analysis_scatter(
    csv_path: str = "static/tweets-processed.csv",
    x_var: str = "Toxic",
    y_var: str = "Insult",
    x_threshold: float = None,
    y_threshold: float = None,
    sample_size: int = None
):
    """
    Creates a scatter plot of two text analysis variables colored by btc_delta_24h.
    Includes dynamic threshold lines to create quadrants for deeper analysis.
    
    Parameters:
    -----------
    csv_path : str
        Path to the processed tweets CSV file
    x_var : str
        Name of the text analysis variable for x-axis (default: "Toxic")
    y_var : str
        Name of the text analysis variable for y-axis (default: "Insult")
    x_threshold : float, optional
        Threshold value for x-axis to draw vertical reference line
    y_threshold : float, optional
        Threshold value for y-axis to draw horizontal reference line
    sample_size : int, optional
        Number of random entries to sample from the dataset. If None, uses all data.
    
    Returns:
    --------
    alt.Chart
        Altair chart object ready to be displayed in Streamlit
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Select relevant columns
    plot_df = df[[x_var, y_var, 'btc_delta_24h']].copy()
    
    # Sample random entries if sample_size is specified
    if sample_size is not None and sample_size > 0:
        # Ensure we don't sample more than available
        n_samples = min(sample_size, len(plot_df))
        plot_df = plot_df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    
    # Create color condition: red if btc_delta_24h <= 0, green otherwise
    plot_df['color'] = plot_df['btc_delta_24h'].apply(
        lambda x: 'red' if x <= 0 else 'green'
    )
    
    # Create the scatter plot
    scatter = alt.Chart(plot_df).mark_circle(
        size=50,
        opacity=0.6
    ).encode(
        x=alt.X(
            x_var,
            title=x_var,
            scale=alt.Scale(zero=False)
        ),
        y=alt.Y(
            y_var,
            title=y_var,
            scale=alt.Scale(zero=False)
        ),
        color=alt.Color(
            'color',
            title='Delta BTC 24h',
            scale=alt.Scale(
                domain=['red', 'green'],
                range=['#FF0000', '#00FF00']
            ),
            legend=alt.Legend(
                title='Delta BTC 24h',
                values=['≤ 0', '> 0']
            )
        ),
        tooltip=[x_var, y_var, 'btc_delta_24h']
    ).properties(
        width=600,
        height=400,
        title=f'{x_var} vs {y_var} (coloreado por Delta BTC 24h)' + 
              (f' - {len(plot_df)} muestras' if sample_size else '')
    )
    
    # Build the chart layers
    chart_layers = [scatter]
    
    # Add vertical reference line for x_threshold if provided
    if x_threshold is not None:
        x_line = alt.Chart(pd.DataFrame({'threshold': [x_threshold]})).mark_rule(
            color='lightblue',
            strokeWidth=2,
            strokeDash=[5, 5],
            opacity=0.7
        ).encode(
            x=alt.X('threshold:Q', title=x_var)
        )
        chart_layers.append(x_line)
    
    # Add horizontal reference line for y_threshold if provided
    if y_threshold is not None:
        y_line = alt.Chart(pd.DataFrame({'threshold': [y_threshold]})).mark_rule(
            color='lightblue',
            strokeWidth=2,
            strokeDash=[5, 5],
            opacity=0.7
        ).encode(
            y=alt.Y('threshold:Q', title=y_var)
        )
        chart_layers.append(y_line)
    
    # Combine all layers
    chart = alt.layer(*chart_layers).resolve_scale().interactive()
    
    return chart, plot_df


def load_btc_daily_data(csv_path: str = "static/btc_15m_data_2018_to_2025.csv"):
    """
    Load BTC 15-minute data and filter to get one entry per day at 16:00 (4 PM).
    
    Parameters:
    -----------
    csv_path : str
        Path to the BTC 15-minute data CSV file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with one entry per day at 16:00
    """
    df = pd.read_csv(csv_path)
    
    # Parse Open time to datetime
    df['Open time'] = pd.to_datetime(df['Open time'])
    
    # Filter to entries at 16:00 (hour = 16)
    df['hour'] = df['Open time'].dt.hour
    df_16h = df[df['hour'] == 16].copy()
    
    # Get one entry per day (take the first one if multiple at 16:00)
    df_16h['date'] = df_16h['Open time'].dt.date
    df_daily = df_16h.groupby('date').first().reset_index()
    
    # Sort by date
    df_daily = df_daily.sort_values('Open time').reset_index(drop=True)
    
    # Keep date as date object for easier comparison
    df_daily['date_only'] = df_daily['date']
    
    return df_daily


def class_to_slope(class_pred: int, base_price: float):
    """
    Map predicted class to slope value for BTC price prediction.
    
    Parameters:
    -----------
    class_pred : int
        Predicted class (0-3)
    base_price : float
        Base BTC price to calculate slope from
    
    Returns:
    --------
    float
        Slope value (price change per day)
    """
    # Map classes to representative delta_24h values
    # These represent the expected percentage change
    class_deltas = {
        0: -0.06,  # btc_delta_24h < -0.06
        1: -0.03,  # -0.06 ≤ btc_delta_24h ≤ 0 (middle of range)
        2: 0.001,  # 0 < btc_delta_24h ≤ 0.002 (middle of range)
        3: 0.01,   # btc_delta_24h > 0.002 (representative positive)
    }
    
    delta = class_deltas.get(class_pred, 0.0)
    # Slope is the price change: price * delta
    slope = base_price * delta
    
    return slope


def plot_btc_price_comparison(
    btc_csv_path: str = "static/btc_15m_data_2018_to_2025.csv",
    tweets_csv_path: str = "static/tweets-processed.csv",
    tweet_date: str = None,
    tweet_text: str = None,
    booster=None,
    preprocessor=None
):
    """
    Plot BTC price over time with predicted vs actual slope comparison for a selected tweet.
    
    Parameters:
    -----------
    btc_csv_path : str
        Path to BTC 15-minute data CSV
    tweets_csv_path : str
        Path to processed tweets CSV
    tweet_date : str
        Date of the selected tweet (format: "YYYY-MM-DD HH:MM:SS")
    tweet_text : str
        Text content of the tweet
    booster : xgb.Booster
        XGBoost model for inference
    preprocessor : sklearn pipeline
        Preprocessor for tweet data
    
    Returns:
    --------
    alt.Chart
        Altair chart with BTC price and slope lines
    """
    # Load BTC daily data
    btc_df = load_btc_daily_data(btc_csv_path)
    
    # Filter data to show only a window around the tweet date (if provided)
    btc_df_filtered = btc_df.copy()
    if tweet_date:
        try:
            tweet_datetime = pd.to_datetime(tweet_date)
            tweet_date_only = tweet_datetime.date()
            
            # Create a window of 7 days before and after the tweet
            start_date = pd.Timestamp(tweet_date_only) - pd.Timedelta(days=7)
            end_date = pd.Timestamp(tweet_date_only) + pd.Timedelta(days=7)
            
            # Filter dataframe
            btc_df_filtered = btc_df[
                (btc_df['Open time'] >= start_date) & 
                (btc_df['Open time'] <= end_date)
            ].copy()
            
            if len(btc_df_filtered) == 0:
                # Fallback to full data if no data in range
                btc_df_filtered = btc_df.copy()
        except:
            # If any error, use full data
            btc_df_filtered = btc_df.copy()
    
    # Create base chart with BTC close prices (filtered data)
    base_chart = alt.Chart(btc_df_filtered).mark_line(
        color='gray',
        strokeWidth=2,
        opacity=0.7
    ).encode(
        x=alt.X('Open time:T', title='Fecha'),
        y=alt.Y('Close:Q', title='Precio BTC', scale=alt.Scale(zero=False)),
        tooltip=['Open time:T', 'Close:Q']
    ).properties(
        height=450,
        title='BTC: Predicción vs Realidad (14 días)'
    )
    
    chart_layers = [base_chart]
    
    # If tweet is selected, add prediction and actual slope lines
    if tweet_date and tweet_text and booster is not None and preprocessor is not None:
        try:
            from lib import process_input
            import xgboost as xgb
            import json
            
            # Parse tweet date
            tweet_datetime = pd.to_datetime(tweet_date)
            tweet_date_only = tweet_datetime.date()
            
            # Load tweets to get additional data
            tweets_df = pd.read_csv(tweets_csv_path)
            tweet_row = tweets_df[tweets_df['date'] == tweet_date].iloc[0] if len(tweets_df[tweets_df['date'] == tweet_date]) > 0 else None
            
            # Prepare input for model
            if tweet_row is not None:
                input_data = {
                    "id": str(tweet_row.get('id', '1')),
                    "text": tweet_text,
                    "device": "web",
                    "favorites": int(tweet_row.get('favorites', 0)),
                    "retweets": int(tweet_row.get('retweets', 0)),
                    "date": tweet_date,
                    "btc_delta_24h": float(tweet_row.get('btc_delta_24h', 0)),
                    "btc_tweet_day": int(tweet_row.get('btc_tweet_day', 0)),
                    "btc_delta_48h": float(tweet_row.get('btc_delta_48h', 0)),
                    "btc_24h_after": float(tweet_row.get('btc_24h_after', 0)),
                    "btc_48h_after": float(tweet_row.get('btc_48h_after', 0))
                }
            else:
                input_data = {
                    "id": "1",
                    "text": tweet_text,
                    "device": "web",
                    "favorites": 0,
                    "retweets": 0,
                    "date": tweet_date,
                    "btc_delta_24h": 0,
                    "btc_tweet_day": 0,
                    "btc_delta_48h": 0,
                    "btc_24h_after": 0,
                    "btc_48h_after": 0
                }
            
            # Process input and predict
            data = process_input(json.dumps(input_data), preprocessor)
            feature_names = preprocessor.named_steps["column_transformer"].get_feature_names_out().tolist()
            dmatrix = xgb.DMatrix(data, feature_names=feature_names)
            pred_class = int(booster.predict(dmatrix)[0])
            
            # Get BTC close price for tweet day
            btc_tweet_day = btc_df[btc_df['date_only'] == tweet_date_only]
            
            if len(btc_tweet_day) == 0:
                # Try to find closest date
                date_diffs = [(d - tweet_date_only).days for d in btc_df['date_only']]
                date_diffs = pd.Series([abs(d) for d in date_diffs], index=btc_df.index)
                closest_idx = date_diffs.idxmin()
                btc_tweet_day = btc_df.iloc[[closest_idx]]
            
            if len(btc_tweet_day) > 0:
                tweet_close_price = float(btc_tweet_day.iloc[0]['Close'])
                tweet_datetime_value = pd.Timestamp(btc_tweet_day.iloc[0]['Open time'])
                
                # Add a vertical line at the tweet time
                tweet_marker_df = pd.DataFrame({'tweet_time': [tweet_datetime_value]})
                tweet_marker = alt.Chart(tweet_marker_df).mark_rule(
                    color='blue',
                    strokeWidth=2,
                    strokeDash=[10, 5],
                    opacity=0.5
                ).encode(
                    x=alt.X('tweet_time:T', title='Fecha')
                )
                chart_layers.append(tweet_marker)
                
                # Add a point at the tweet moment
                tweet_point_df = pd.DataFrame({
                    'time': [tweet_datetime_value],
                    'price': [tweet_close_price],
                    'label': ['📍 Tweet']
                })
                tweet_point = alt.Chart(tweet_point_df).mark_point(
                    color='blue',
                    size=200,
                    filled=True
                ).encode(
                    x=alt.X('time:T', title='Fecha'),
                    y=alt.Y('price:Q', title='Precio BTC (Close)'),
                    tooltip=['label:N', 'time:T', 'price:Q']
                )
                chart_layers.append(tweet_point)
                
                # Calculate predicted slope
                predicted_slope = class_to_slope(pred_class, tweet_close_price)
                
                # Get next day's close price for actual slope
                next_date = pd.Timestamp(tweet_date_only) + pd.Timedelta(days=1)
                next_date_only = next_date.date()
                btc_next_day = btc_df[btc_df['date_only'] == next_date_only]
                if len(btc_next_day) > 0:
                    next_close_price = float(btc_next_day.iloc[0]['Close'])
                    actual_slope = next_close_price - tweet_close_price
                    next_datetime_value = pd.Timestamp(btc_next_day.iloc[0]['Open time'])
                    
                    # Create line data for predicted slope (y = mx + b)
                    # Using tweet_close_price as base (b) and slope as m
                    line_points = pd.DataFrame({
                        'x': [tweet_datetime_value, next_datetime_value],
                        'y_predicted': [
                            tweet_close_price,
                            tweet_close_price + predicted_slope
                        ],
                        'y_actual': [
                            tweet_close_price,
                            next_close_price
                        ]
                    })
                    
                    # Predicted slope line
                    pred_line = alt.Chart(line_points).mark_line(
                        color='red',
                        strokeWidth=4,
                        strokeDash=[5, 5]
                    ).encode(
                        x=alt.X('x:T', title='Fecha'),
                        y=alt.Y('y_predicted:Q', title='Precio BTC (Close)')
                    )
                    
                    # Actual slope line
                    actual_line = alt.Chart(line_points).mark_line(
                        color='green',
                        strokeWidth=4,
                        strokeDash=[2, 2]
                    ).encode(
                        x=alt.X('x:T', title='Fecha'),
                        y=alt.Y('y_actual:Q', title='Precio BTC (Close)')
                    )
                    
                    chart_layers.extend([pred_line, actual_line])
        except Exception as e:
            print(f"Error plotting tweet prediction: {e}")
    
    # Combine all layers
    chart = alt.layer(*chart_layers).resolve_scale(
        x='shared',
        y='shared'
    ).interactive()
    
    return chart


def precompute_predictions(
    tweets_csv_path: str = "static/tweets-processed.csv",
    output_path: str = "static/tweets-with-predictions.csv",
    booster=None,
    preprocessor=None,
    batch_size: int = 100
):
    """
    Pre-compute predictions for all tweets and save to CSV.
    
    Parameters:
    -----------
    tweets_csv_path : str
        Path to processed tweets CSV file
    output_path : str
        Path to save tweets with predictions
    booster : xgb.Booster
        XGBoost model for inference
    preprocessor : sklearn pipeline
        Preprocessor for tweet data
    batch_size : int
        Number of tweets to process at once
    """
    from lib import process_input
    import xgboost as xgb
    import json
    
    # Load all tweets
    df = pd.read_csv(tweets_csv_path)
    print(f"Pre-computing predictions for {len(df)} tweets...")
    
    predictions = []
    
    # Process in batches for progress tracking
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        for idx, row in batch.iterrows():
            try:
                input_data = {
                    "id": str(row.get('id', idx)),
                    "text": str(row.get('text', '')),
                    "device": "web",
                    "favorites": int(row.get('favorites', 0)),
                    "retweets": int(row.get('retweets', 0)),
                    "date": str(row.get('date', '')),
                    "btc_delta_24h": float(row.get('btc_delta_24h', 0)),
                    "btc_tweet_day": int(row.get('btc_tweet_day', 0)),
                    "btc_delta_48h": float(row.get('btc_delta_48h', 0)),
                    "btc_24h_after": float(row.get('btc_24h_after', 0)),
                    "btc_48h_after": float(row.get('btc_48h_after', 0))
                }
                
                data = process_input(json.dumps(input_data), preprocessor)
                feature_names = preprocessor.named_steps["column_transformer"].get_feature_names_out().tolist()
                dmatrix = xgb.DMatrix(data, feature_names=feature_names)
                pred_class = int(booster.predict(dmatrix)[0])
                predictions.append(pred_class)
            except Exception as e:
                predictions.append(None)
        
        if (i + batch_size) % 1000 == 0:
            print(f"Processed {min(i + batch_size, len(df))} / {len(df)} tweets...")
    
    # Add predictions to dataframe
    df['predicted_class'] = predictions
    df = df[df['predicted_class'].notna()].copy()
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} tweets with predictions to {output_path}")
    
    return df


def plot_variable_by_class_boxplot(
    tweets_csv_path: str = "static/tweets-processed.csv",
    variable: str = "Toxic",
    sample_size: int = 1000
):
    """
    Create boxplots of a selected variable grouped by actual classes based on btc_delta_24h.
    
    Parameters:
    -----------
    tweets_csv_path : str
        Path to processed tweets CSV file
    variable : str
        Variable to plot (must be in the CSV)
    sample_size : int
        Number of random entries to sample
    
    Returns:
    --------
    alt.Chart
        Altair chart with boxplots grouped by class
    """
    from lib import CLASS_RANGES
    
    # Load tweets
    df = pd.read_csv(tweets_csv_path)
    
    # Sample random entries
    if sample_size is not None and sample_size > 0:
        n_samples = min(sample_size, len(df))
        df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    
    # Calculate class from btc_delta_24h
    def calculate_class(btc_delta):
        """Calculate class based on btc_delta_24h value."""
        if pd.isna(btc_delta):
            return None
        if btc_delta < -0.06:
            return 0
        elif -0.06 <= btc_delta <= 0:
            return 1
        elif 0 < btc_delta <= 0.002:
            return 2
        else:  # btc_delta > 0.002
            return 3
    
    df['actual_class'] = df['btc_delta_24h'].apply(calculate_class)
    
    # Filter out rows with no class
    df = df[df['actual_class'].notna()].copy()
    
    # Check if variable exists
    if variable not in df.columns:
        raise ValueError(f"Variable '{variable}' not found in dataset")
    
    # Create class labels with ranges
    class_labels_map = {
        0: f"Clase 0: {CLASS_RANGES.get(0, 'Desconocido')}",
        1: f"Clase 1: {CLASS_RANGES.get(1, 'Desconocido')}",
        2: f"Clase 2: {CLASS_RANGES.get(2, 'Desconocido')}",
        3: f"Clase 3: {CLASS_RANGES.get(3, 'Desconocido')}"
    }
    
    df['class_label'] = df['actual_class'].apply(
        lambda x: class_labels_map.get(x, f"Clase {x}: Desconocido")
    )
    
    # Define sort order based on actual labels
    sort_order = [class_labels_map[i] for i in range(4)]
    
    # Create selection for zoom
    zoom = alt.selection_interval(bind='scales')
    
    # Create boxplot with zoom capability
    chart = alt.Chart(df).mark_boxplot(
        size=50,
        extent='min-max'
    ).encode(
        x=alt.X(
            'class_label:N',
            title='Clase Real',
            sort=sort_order
        ),
        y=alt.Y(
            f'{variable}:Q',
            title=variable,
            scale=alt.Scale(zero=False)
        ),
        color=alt.Color(
            'class_label:N',
            legend=None
        ),
        tooltip=['actual_class:Q', f'{variable}:Q', 'class_label:N', 'btc_delta_24h:Q']
    ).add_selection(
        zoom
    ).properties(
        width=800,
        height=400,
        title=f'Distribución de {variable} por Clase Real'
    ).interactive()
    
    return chart

